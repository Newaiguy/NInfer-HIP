#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/storage/kv_archive.h"
#include "models/qwen3_5/program/storage/kv_retrieval.h"
#include "models/qwen3_5/execution/kv_window.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

bool ProgramImpl::valid_continuation(const ContinuationHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < continuation_capacity &&
           ContractAccess::epoch(handle) == continuation_slots[index].generation &&
           continuation_slots[index].role == ContinuationSlotRole::Catalogued;
}

bool ProgramImpl::valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < shared_prefix_capacity &&
           ContractAccess::epoch(handle) == shared_prefix_slots[index].generation &&
           shared_prefix_slots[index].role == SharedPrefixSlotRole::Catalogued;
}

bool ProgramImpl::valid_capture_offer(const CaptureOffer& offer) const noexcept {
    if (ContractAccess::owner(offer) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    if (lane >= max_concurrency || ContractAccess::epoch(offer) != lane_epochs[lane] ||
        (requests[lane].lifecycle != Lifecycle::Prefilling &&
         requests[lane].lifecycle != Lifecycle::Active) ||
        !requests[lane].prefill) {
        return false;
    }
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    return prefill.pending_capture_offer != 0 &&
           prefill.pending_capture_offer == ContractAccess::id(offer) &&
           prefill.next_capture < prefill.capture_groups.size() &&
           prefill.cursor == prefill.capture_groups[prefill.next_capture].frontier;
}

bool ProgramImpl::materialization_pins(std::uint32_t index,
                                       std::uint64_t generation) const noexcept {
    const MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) { return false; }
    const MaterializationTransaction& transaction = *transaction_ptr;
    if (transaction.has_source && transaction.source_index == index &&
        transaction.source_generation == generation) {
        return true;
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim] && transaction.victim_indices[victim] == index &&
            transaction.victim_generations[victim] == generation) {
            return true;
        }
    }
    return false;
}

bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

std::optional<std::uint32_t> ProgramImpl::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

bool ProgramImpl::can_release_continuation_slot_strict(std::uint32_t index) const {
    if (index >= continuation_capacity || !state_store || !text_kv_addresses || !text_kv_pages ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        return false;
    }
    const SequenceState& sequence = continuation_states[index];
    if (sequence.state.fork_pending || !sequence.shared_prefix_references.empty() || !sequence.kv ||
        !text_kv_addresses->can_release(sequence.kv->text)) {
        return false;
    }
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages ||
            !backend_kv_addresses->can_release(*sequence.kv->backend)) {
            return false;
        }
    }

    const auto validate_state = [&](StateImageHandle handle, bool release_object) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
            return false;
        }
        return !release_object || total != owned ||
               state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
        if (handle == sequence.state.read || handle == sequence.state.write ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
            return true;
        }
        for (std::size_t prior = 0; prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { return true; }
        }
        return false;
    };

    if (sequence.endpoint_valid) {
        if (!validate_state(sequence.state.read, !sequence.state.read_has_external_owner() ||
                                                     sequence.state.read == sequence.state.write)) {
            return false;
        }
        if (sequence.state.write != sequence.state.read &&
            !validate_state(sequence.state.write, true)) {
            return false;
        }
    } else if (state_store->valid(sequence.state.read) ||
               state_store->valid(sequence.state.write) || sequence.state.borrows_read()) {
        return false;
    }
    if (sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read &&
        *sequence.rewrite_state != sequence.state.write &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t anchor = 0; anchor < sequence.long_anchors.size(); ++anchor) {
        const StateImageHandle handle = sequence.long_anchors[anchor].state;
        if (!repeated_before_anchor(anchor, handle) && !validate_state(handle, true)) {
            return false;
        }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated = handle == sequence.state.read || handle == sequence.state.write ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_continuation_slot_strict(std::uint32_t index) noexcept {
    try {
        if (!can_release_continuation_slot_strict(index)) { std::terminate(); }
    } catch (...) { std::terminate(); }
    SequenceState& sequence = continuation_states[index];
    release_sequence_kv_strict(sequence);
    release_sequence_state_strict(sequence);
    retire_continuation_slot(index);
}

void ProgramImpl::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    SequenceState& sequence = continuation_states[index];
    if (std::getenv("NINFER_KV_RELEASE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-slot] retire index=%u seq=%p\n", index,
                     static_cast<const void*>(&sequence));
        std::fflush(stderr);
    }
    release_active_shared_references(sequence);
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
}

void ProgramImpl::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_digests.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    sequence.rewrite_checkpoint      = {};
    sequence.rebuild_work            = {};
    sequence.rebuild_tail_begin      = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

detail::PhysicalResources
ProgramImpl::sequence_exclusive_state_resources(const SequenceState& sequence) const {
    if (!state_store) {
        throw std::logic_error("sequence StateImage resources have no physical store");
    }
    detail::PhysicalResources out;
    std::array<StateImageHandle, 4> states{};
    std::uint32_t state_count = 0;
    const auto add_state      = [&](StateImageHandle handle) {
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { return; }
        for (std::uint32_t index = 0; index < state_count; ++index) {
            if (states[index] == handle) { return; }
        }
        states[state_count++]                 = handle;
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    };
    const bool has_read_state  = sequence.state.read.valid();
    const bool has_write_state = sequence.state.write.valid();
    if (has_read_state != has_write_state) {
        throw std::logic_error("sequence owner has a partial primary StateImage pair");
    }
    if (sequence.state.borrows_read() &&
        (!sequence.state.fork_pending || sequence.state.read == sequence.state.write)) {
        throw std::logic_error("sequence has an invalid borrowed StateImage source");
    }
    if (has_read_state) {
        if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
            add_state(sequence.state.read);
        }
        add_state(sequence.state.write);
    }
    if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale long-anchor StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { continue; }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(state_count, states.size());
             ++index) {
            if (states[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (seen) { continue; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SequenceState& sequence) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("sequence owner resources have no physical stores");
    }
    detail::PhysicalResources out = sequence_exclusive_state_resources(sequence);

    {
        if (!sequence.kv) { throw std::logic_error("sequence owner has no KV address bundle"); }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale KV address space"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                // A shared logical page contributes to aggregate occupancy once. Releasing this
                // address cannot free either replica while another address still references it,
                // so it is not part of this owner's exact transition effect.
                if (pages.address_references(logical) > 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("resident Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
            if (addresses.active(address)) {
                const std::uint32_t mapped      = addresses.mapped_pages(address);
                const std::uint32_t entitlement = addresses.entitlement(address);
                if (entitlement < mapped ||
                    entitlement - mapped >
                        std::numeric_limits<std::uint32_t>::max() - device_pages) {
                    throw std::logic_error("owner active KV entitlement is inconsistent");
                }
                device_pages += entitlement - mapped;
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.device.main_kv_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SharedPrefixState& shared) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("shared owner resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        if (!shared.kv || !shared.identity || !state_store->valid(shared.state)) {
            throw std::logic_error("shared prefix has incomplete resident physical state");
        }
        if (state_store->checkpoint_references(shared.state) == 0) {
            throw std::logic_error("shared prefix StateImage has no checkpoint reference");
        }
        const StateReplicaResidency residency = state_store->residency(shared.state);
        if (state_store->checkpoint_references(shared.state) == 1) {
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale shared KV address"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) != 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("shared Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, out.device.main_kv_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing shared Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources ProgramImpl::physical_occupancy() const noexcept {
    detail::PhysicalResources out;
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) { ++out.device.active_lanes; }
    }
    if (state_store) {
        out.device.state_slots = state_store->device_occupied();
        out.host.state_slots   = state_store->host_occupied();
    }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        out.device.main_kv_pages     = pool.allocated_pages() + pool.reserved_pages();
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device.backend_kv_pages  = pool.allocated_pages() + pool.reserved_pages();
    }
    if (host_kv_arena) { out.host.kv_bytes = host_kv_arena->occupied_bytes(); }
    return out;
}

detail::PhysicalResources
ProgramImpl::materialization_deficit(const ResourceCandidateState& admission) const {
    // Pressure is relative to this candidate's real peak. Treating every dimension as scarce
    // would forbid Device-to-Host demotion even when Host capacity is available.
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), admission.demand.physical_peak_additional);
    return positive_resource_difference(required, admission_capacity());
}

detail::PhysicalResources
ProgramImpl::guided_materialization_deficit(const ResourceCandidateState& admission,
                                            const detail::PhysicalDelta& pressure) const {
    // Pressure acts on the candidate's complete peak, not on its already-clamped deficit.  Applying
    // a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even
    // when the arena has ample slack and steer the heuristic toward unnecessary destruction.
    const detail::PhysicalResources projected_peak = positive_resource_difference(
        checked_resource_sum(admission.demand.physical_peak_additional, pressure.added),
        pressure.removed);
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), projected_peak);
    return positive_resource_difference(required, admission_capacity());
}

bool ProgramImpl::physical_peak_fits(detail::PhysicalResources peak) const noexcept {
    const detail::PhysicalResources occupied = physical_occupancy();
    const detail::PhysicalResources limits   = admission_capacity();
    const auto fits_u32 = [](std::uint32_t used, std::uint32_t added, std::uint32_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    const auto fits_size = [](std::size_t used, std::size_t added, std::size_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    return fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                    limits.device.active_lanes) &&
           fits_u32(occupied.device.state_slots, peak.device.state_slots,
                    limits.device.state_slots) &&
           fits_u32(occupied.device.main_kv_pages, peak.device.main_kv_pages,
                    limits.device.main_kv_pages) &&
           fits_u32(occupied.device.backend_kv_pages, peak.device.backend_kv_pages,
                    limits.device.backend_kv_pages) &&
           fits_u32(occupied.host.state_slots, peak.host.state_slots, limits.host.state_slots) &&
           fits_size(occupied.host.kv_bytes, peak.host.kv_bytes, limits.host.kv_bytes);
}

StateImageHandle
ProgramImpl::selected_state(const SequenceState& sequence, ReusePath reuse,
                            std::optional<runtime::CheckpointRef> checkpoint) const {
    if (reuse == ReusePath::PrivateEndpoint) {
        if (!sequence.endpoint_valid || !state_store->valid(sequence.state.read)) {
            throw std::logic_error("private endpoint StateImage is stale");
        }
        return sequence.state.read;
    }
    if (is_rewrite_checkpoint_restore(reuse) && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state)) {
        return *sequence.rewrite_state;
    }
    if (reuse == ReusePath::PrivateLongAnchor) {
        if (!checkpoint || checkpoint->kind != runtime::CheckpointKind::LongAnchor) {
            throw std::logic_error("long-anchor materialization has no selected checkpoint");
        }
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint->frontier &&
                                                    candidate.ordinal == checkpoint->ordinal;
                                         });
        if (anchor != sequence.long_anchors.end() && state_store->valid(anchor->state)) {
            return anchor->state;
        }
    }
    throw std::logic_error("materialization path has no selected StateImage");
}

std::uint32_t
ProgramImpl::selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                                RewriteCheckpointDisposition rewrite_disposition,
                                                std::optional<runtime::CheckpointRef> checkpoint,
                                                std::uint32_t reuse_base) const {
    const StateImageHandle selected   = selected_state(sequence, reuse, checkpoint);
    std::uint32_t consumed_references = 0;
    if (is_rewrite_checkpoint_restore(reuse) &&
        rewrite_disposition != RewriteCheckpointDisposition::RetainExisting) {
        if (!sequence.rewrite_state || *sequence.rewrite_state != selected) {
            throw std::logic_error("selected rewrite StateImage is unavailable");
        }
        consumed_references = 1;
    } else if (reuse == ReusePath::PrivateEndpoint &&
               rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
               sequence.rewrite_state && *sequence.rewrite_state == selected) {
        consumed_references = 1;
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier > reuse_base && anchor.state == selected) {
            if (consumed_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("consumed StateImage reference inventory overflow");
            }
            ++consumed_references;
        }
    }
    const std::uint32_t references = state_store->checkpoint_references(selected);
    if (consumed_references > references) {
        throw std::logic_error("selected StateImage reference inventory is inconsistent");
    }
    return consumed_references;
}

bool ProgramImpl::selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                               RewriteCheckpointDisposition rewrite_disposition,
                                               std::optional<runtime::CheckpointRef> checkpoint,
                                               std::uint32_t reuse_base) const {
    const StateImageHandle selected = selected_state(sequence, reuse, checkpoint);
    return state_store->checkpoint_references(selected) !=
           selected_state_consumed_references(sequence, reuse, rewrite_disposition, checkpoint,
                                              reuse_base);
}

bool ProgramImpl::can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                const RewriteCheckpointSpec& desired,
                                                const SequenceState& sequence, ReusePath reuse,
                                                std::uint32_t reuse_base) const {
    if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
        !state_store->valid(*sequence.rewrite_state) ||
        !qwen3_5::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                         sequence.rewrite_checkpoint.frontier)) {
        return false;
    }
    if (sequence.rewrite_checkpoint.frontier == desired.frontier) { return true; }
    return is_rewrite_checkpoint_restore(reuse) &&
           sequence.rewrite_checkpoint.frontier == reuse_base && desired.frontier <= reuse_base;
}

std::uint32_t ProgramImpl::device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.device_resident(addresses.logical_page(address, page))) { ++resident; }
    }
    return resident;
}

std::uint32_t ProgramImpl::shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t shared = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.address_references(addresses.logical_page(address, page)) <= 1) { continue; }
        if (page + 1U == required && frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0) {
            continue;
        }
        ++shared;
    }
    return shared;
}

std::uint32_t ProgramImpl::shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                         KVAddressSpaceHandle address,
                                                         std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && pages.device_resident(logical)) { ++resident; }
    }
    return resident;
}

bool ProgramImpl::partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address,
                                            std::uint32_t frontier) const {
    if (frontier == 0 || frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
        return false;
    }
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
    return pages.address_references(tail) > 1 || !pages.device_resident(tail);
}

std::uint32_t
ProgramImpl::missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t missing = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && !pages.device_resident(logical)) { ++missing; }
    }
    return missing;
}

std::size_t ProgramImpl::host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                              KVAddressSpaceHandle address,
                                              std::uint32_t frontier) const noexcept {
    if (!host_kv_extents) { return 0; }
    try {
        const LogicalKVPageStore& pages =
            (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
        const std::uint32_t required_pages = kv_pages_for_frontier(frontier);
        if (required_pages > addresses.mapped_pages(address)) { return 0; }
        std::size_t bytes = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) > 1) { continue; }
            if (!pages.host_resident(logical)) { continue; }
            if (page + 1U == required_pages &&
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0 &&
                partial_tail_cow_required(addresses, address, frontier)) {
                continue;
            }
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t selected_columns =
                std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (selected_columns != pages.committed_columns(logical)) {
                // A destructive private rewrite changes this tail page's content epoch, so its
                // old Host replica cannot remain part of the active entitlement.
                continue;
            }
            const std::size_t stride =
                host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
            bytes += stride;
        }
        return bytes;
    } catch (...) { return 0; }
}

qwen3_5::CheckpointSummary
ProgramImpl::checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                                StateImageHandle state, runtime::PrefillWork rebuild_work) const {
    if (!sequence.kv) { throw std::logic_error("checkpoint summary has no KV address space"); }
    if (checkpoint.frontier == 0) {
        throw std::logic_error("checkpoint summary has an empty frontier");
    }
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint summary has a stale StateImage");
    }
    const StateReplicaResidency state_location = state_store->residency(state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("checkpoint StateImage has no published replica");
    }
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? checkpoint.frontier - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? checkpoint.frontier
                                                            : 0U;
    const std::uint32_t identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                       (static_cast<std::uint32_t>(proposal_head) << 8U) |
                                       (static_cast<std::uint32_t>(kv_storage) << 16U);
    return qwen3_5::CheckpointSummary{
        .ref   = checkpoint,
        .scope = runtime::CheckpointScope::Private,
        .shortlist_key =
            {
                .digests      = sequence.prefix_digests.at(checkpoint.frontier),
                .frontier     = checkpoint.frontier,
                .identity_tag = identity_tag,
            },
        .state_residency = residency,
        .required_kv =
            {
                .main_frontier    = checkpoint.frontier,
                .backend_frontier = backend_frontier,
                .main_pages       = kv_pages_for_frontier(checkpoint.frontier),
                .backend_pages    = kv_pages_for_frontier(backend_frontier),
            },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
    };
}

qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const SequenceState& sequence) const {
    qwen3_5::ContinuationSummary summary;
    summary.long_anchors.reserve(sequence.long_anchors.size());
    populate_continuation_summary(sequence, summary);
    return summary;
}

void ProgramImpl::populate_continuation_summary(const SequenceState& sequence,
                                                qwen3_5::ContinuationSummary& summary) const {
    validate_long_anchor_ordinals(sequence.long_anchors,
                                  context_cache.max_long_anchors_per_continuation.value_or(0));
    if (summary.long_anchors.capacity() < sequence.long_anchors.size()) {
        throw std::logic_error("continuation summary backing was not reserved");
    }
    summary.endpoint.reset();
    summary.rewrite.reset();
    summary.long_anchors.clear();
    summary.active_references = 0;
    if (sequence.endpoint_valid) {
        const runtime::CheckpointRef endpoint{
            .kind     = runtime::CheckpointKind::SessionEndpoint,
            .frontier = sequence.execution_frontier,
        };
        runtime::PrefillWork endpoint_work = sequence.rebuild_work;
        summary.endpoint =
            checkpoint_summary(sequence, endpoint, sequence.state.read, endpoint_work);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        const runtime::CheckpointRef rewrite{
            .kind     = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier = sequence.rewrite_checkpoint.frontier,
        };
        summary.rewrite = checkpoint_summary(sequence, rewrite, *sequence.rewrite_state,
                                             sequence.rewrite_checkpoint.rebuild_work);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        summary.long_anchors.push_back(
            checkpoint_summary(sequence,
                               runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                                      .frontier = anchor.frontier,
                                                      .ordinal  = anchor.ordinal},
                               anchor.state, anchor.rebuild_work));
    }
    if (!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) {
        throw std::logic_error("private continuation has no checkpoint");
    }
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        const std::size_t index = static_cast<std::size_t>(&sequence - begin);
        summary.active_references =
            continuation_slots[index].role == ContinuationSlotRole::Active ? 1U : 0U;
    }
}

qwen3_5::SharedPrefixSummary
ProgramImpl::shared_prefix_summary(const SharedPrefixState& shared) const {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        !state_store->valid(shared.state)) {
        throw std::logic_error("shared-prefix summary source is incomplete");
    }
    const StateReplicaResidency state_location = state_store->residency(shared.state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("shared-prefix StateImage has no published replica");
    }
    return qwen3_5::SharedPrefixSummary{
        .checkpoint =
            {
                .ref =
                    {
                        .kind     = runtime::CheckpointKind::SharedStablePrefix,
                        .frontier = shared.frontier,
                    },
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shared.identity->shortlist_key,
                .state_residency = residency,
                .required_kv =
                    {
                        .main_frontier    = shared.frontier,
                        .backend_frontier = shared.backend_frontier,
                        .main_pages       = kv_pages_for_frontier(shared.frontier),
                        .backend_pages    = kv_pages_for_frontier(shared.backend_frontier),
                    },
                .rebuild_work = validated_rebuild_work(shared.rebuild_work, shared.frontier),
            },
        .active_references = shared.active_references,
    };
}

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

bool ProgramImpl::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    for (std::size_t position = 0; position < sequence.shared_prefix_references.size();
         ++position) {
        const std::uint32_t index = sequence.shared_prefix_references[position];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) {
            return false;
        }
        const std::uint32_t required = static_cast<std::uint32_t>(std::count(
            sequence.shared_prefix_references.begin(),
            sequence.shared_prefix_references.begin() + static_cast<std::ptrdiff_t>(position + 1U),
            index));
        if (shared_prefix_states[index].active_references < required) { return false; }
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state.borrows_read() ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    enum class ForkEndpoint : std::uint8_t { None, Source, Destination };
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    ForkEndpoint fork_endpoint = ForkEndpoint::None) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable &&
             fork_endpoint != ForkEndpoint::Destination)) {
            return false;
        }
        if (!release_object || total != owned) { return true; }
        if (fork_endpoint == ForkEndpoint::Source) {
            return state_store->can_release_source_after_fork_abort(sequence.state.read,
                                                                    sequence.state.write, owned);
        }
        if (fork_endpoint == ForkEndpoint::Destination) {
            return state_store->can_release_destination_after_fork_abort(
                sequence.state.read, sequence.state.write, owned);
        }
        return state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto duplicates_binding = [&](StateImageHandle handle) {
        return handle == sequence.state.read || handle == sequence.state.write;
    };

    if (!validate_state(sequence.state.read,
                        !sequence.state.read_has_external_owner() ||
                            sequence.state.read == sequence.state.write,
                        sequence.state.fork_pending ? ForkEndpoint::Source : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true,
                        sequence.state.fork_pending ? ForkEndpoint::Destination
                                                    : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.rewrite_state && !duplicates_binding(*sequence.rewrite_state) &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
            repeated = sequence.long_anchors[prior].state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImpl::release_active_shared_references_strict(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            std::terminate();
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

bool ProgramImpl::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                = continuation_states.data();
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    release_active_shared_references_strict(sequence);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    return true;
}

void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    // A concurrent resource transaction may pin or inspect these active owners. Engine-wide
    // cleanup aborts that transaction before releasing lanes, preserving the only safe order.
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

void ProgramImpl::clear_lane_best_effort(SequenceState& sequence,
                                         RequestControl& request) noexcept {
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

std::uint32_t ProgramImpl::owned_checkpoint_references(const SequenceState& sequence,
                                                       StateImageHandle state) const noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

bool ProgramImpl::state_exclusive_to_sequence(const SequenceState& sequence,
                                              StateImageHandle state) const noexcept {
    if (!state_store || !state_store->valid(state)) { return false; }
    return state_store->checkpoint_references(state) ==
           owned_checkpoint_references(sequence, state);
}

void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}

void ProgramImpl::reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots) {
    const std::uint32_t owned = sequence_exclusive_state_resources(sequence).device.state_slots;
    if (slots == 0 || owned > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (owned == slots) { return; }
    if (slots - owned != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = state_store->reserve_destination();
    if (!reserved) { throw std::bad_alloc(); }
    sequence.reserved_state = *reserved;
    if (sequence_exclusive_state_resources(sequence).device.state_slots != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    const bool external_source         = sequence.state.read_has_external_owner();
    state_store->commit_fork(source, destination);
    sequence.state = ActiveStateBinding{.read = destination, .write = destination};
    if (!external_source && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    refresh_state_views(sequence);
}

bool ProgramImpl::has_unsettled_state_fork() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (continuation < continuation_capacity &&
            continuation_states[continuation].state.fork_pending) {
            return true;
        }
    }
    return false;
}

void ProgramImpl::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto duplicates_binding = [&](StateImageHandle handle) {
            return handle == sequence.state.read || handle == sequence.state.write;
        };

        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state.read_has_external_owner());
        }
        if (sequence.rewrite_state) {
            release_if_unreferenced(*sequence.rewrite_state,
                                    !duplicates_binding(*sequence.rewrite_state));
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
                repeated = sequence.long_anchors[prior].state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store || sequence.state.fork_pending) { fail(); }

    try {
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
            if ((sequence.endpoint_valid &&
                 (handle == sequence.state.read || handle == sequence.state.write)) ||
                (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
                return true;
            }
            for (std::size_t prior = 0; prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { return true; }
            }
            return false;
        };

        if (sequence.endpoint_valid) {
            release_if_unreferenced(sequence.state.write, true);
            if (sequence.state.read != sequence.state.write) {
                release_if_unreferenced(sequence.state.read,
                                        !sequence.state.read_has_external_owner());
            }
        }
        if (sequence.rewrite_state) {
            const StateImageHandle handle = *sequence.rewrite_state;
            const bool duplicates_endpoint =
                sequence.endpoint_valid &&
                (handle == sequence.state.read || handle == sequence.state.write);
            release_if_unreferenced(handle, !duplicates_endpoint);
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            release_if_unreferenced(handle, !repeated_before_anchor(index, handle));
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = sequence.endpoint_valid &&
                            (handle == sequence.state.read || handle == sequence.state.write);
            repeated = repeated || (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }

    try {
        if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
            state_store->checkpoint_references(*sequence.rewrite_state) != 0) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            if (state_store->valid(anchor.state) &&
                state_store->checkpoint_references(anchor.state) != 0) {
                state_store->release_checkpoint_reference(anchor.state);
            }
        }
    } catch (...) {}

    const auto releasable = [&](StateImageHandle handle) { return state_store->valid(handle); };
    if (releasable(sequence.state.write)) { (void)state_store->release(sequence.state.write); }
    if (!sequence.state.read_has_external_owner() && sequence.state.read != sequence.state.write &&
        releasable(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    if (sequence.rewrite_state) {
        const StateImageHandle handle = *sequence.rewrite_state;
        const bool duplicates_binding =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read);
        if (!duplicates_binding && releasable(handle)) { (void)state_store->release(handle); }
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t previous = 0; !duplicate && previous < index; ++previous) {
            duplicate = sequence.long_anchors[previous].state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            duplicate = duplicate || anchor.state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

void ProgramImpl::release_active_shared_references(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            continue;
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

const qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImpl::resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                                 std::uint32_t backend_pages) {
    if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-op] resize text=%u backend=%u\n", text_pages, backend_pages);
        std::fflush(stderr);
    }
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    text_kv_addresses->resize_entitlement(sequence.kv->text, text_pages);
    if (sequence.kv->backend) {
        backend_kv_addresses->resize_entitlement(*sequence.kv->backend, backend_pages);
    }
}

void ProgramImpl::bind_sequence_kv(SequenceState& sequence) {
    if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-op] bind active=%d\n",
                     text_kv_addresses->active(sequence.kv->text) ? 1 : 0);
        std::fflush(stderr);
    }
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

void ProgramImpl::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

namespace {
// Leg B windows can pin a leading "sink" range (system prompt / task statement) that survives
// every slide, like the sink blocks in kvmem. Pages are the unit, so the value is rounded down.
std::uint32_t kv_window_sink_tokens() {
    static const std::uint32_t value = [] {
        const char* env          = std::getenv("NINFER_SINK_TOKENS");
        // A pinned sink compresses the tail and the recall region has to be carved out of it, so the
        // default stays at zero; --sink-tokens opts into keeping the prefix.
        const unsigned long raw  = env != nullptr ? std::strtoul(env, nullptr, 10) : 0UL;
        return static_cast<std::uint32_t>(raw / kPagedKVPageSize * kPagedKVPageSize);
    }();
    return value;
}

// Leg B / kvmem W3-b: how many of the oldest tail pages are handed over to recalled blocks. The
// region sits directly above the pinned sink, so it is the first thing the next slide would drop.
std::uint32_t kv_window_recall_pages() {
    static const std::uint32_t value = [] {
        const char* env         = std::getenv("NINFER_KV_RECALL_PAGES");
        // 112 pages (7168 tokens) is the validated default: the recall region must be at least as
        // wide as the block the multi-peak selector hands back, otherwise the slide path stops
        // skipping it and archives recalled pages as if they were recent window tokens.
        const unsigned long raw = env != nullptr ? std::strtoul(env, nullptr, 10) : 112UL;
        return static_cast<std::uint32_t>(raw);
    }();
    return value;
}

// True once a recalled block actually landed in the oldest tail slots: from then on those slots
// hold a distant range and the slide path must not index them under window coordinates. Before the
// first swap they are ordinary recent context and have to be archived like everything else.
bool g_kv_recall_armed = false;
bool g_kv_retrieval_done = false;

// Highest absolute token the recall path has already indexed. Each swap re-archives the slots it is
// about to overwrite, and consecutive slides shift those slots by only a few pages, so without this
// high-water mark the same 100+ pages would be copied to host on every single slide.
std::uint32_t g_kv_archive_watermark = 0;
}  // namespace

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens) {
    if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-op] map main=%u backend=%u\n", main_tokens, backend_tokens);
        std::fflush(stderr);
    }
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    // Resident-window mode (Leg B): the Main KV address space holds only the newest pages, so the
    // execution path addresses it in COMPACT coordinates (absolute position minus the window base);
    // RoPE positions stay absolute. Sliding itself happens at chunk boundaries in ensure_kv_window().
    std::uint32_t compact_tokens  = main_tokens;
    // The speculative (MTP) cache lives in the same windowed address space, so it is compacted the
    // same way; leaving it absolute asked for one page per context token and blew the entitlement.
    std::uint32_t compact_backend = backend_tokens;
    static const bool resident_windowed = [] {
        const char* env = std::getenv("NINFER_RESIDENT_TOKENS");
        return env != nullptr && env[0] != '\0';
    }();
    if (resident_windowed) {
        if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
            std::fprintf(stderr, "[kv-op] map-resident lane=%u active=%d\n",
                         static_cast<unsigned>(sequence.lane),
                         text_kv_addresses->active(sequence.kv->text) ? 1 : 0);
            std::fflush(stderr);
        }
        const std::uint32_t window_base = text_kv_addresses->window_base_tokens(sequence.kv->text);
        const std::uint32_t resident_pages = text_kv_addresses->entitlement(sequence.kv->text);
        const std::uint32_t sink_tokens =
            std::min(kv_window_sink_tokens(), resident_pages * static_cast<std::uint32_t>(kPagedKVPageSize));
        // Absolute -> compact: the retained sink occupies the low slots, so the tail shift is the
        // window base minus the sink that is still mapped in front of it.
        const std::uint32_t shift = window_base == 0 ? 0U : window_base - sink_tokens;
        compact_backend = backend_tokens > shift ? backend_tokens - shift : 0U;
        // Only clamp against the speculative cache when this sequence actually owns one: dereferencing
        // the empty optional handed a garbage handle to entitlement() and every windowed run died with
        // "KV address-space handle is stale" before the first chunk.
        if (sequence.kv->backend) {
            const std::uint32_t backend_capacity =
                backend_kv_addresses->entitlement(*sequence.kv->backend) *
                static_cast<std::uint32_t>(kPagedKVPageSize);
            if (compact_backend > backend_capacity) { compact_backend = backend_capacity; }
        } else {
            compact_backend = 0;
        }
        g_ninfer_kv_window_base_tokens = shift;
        const std::uint32_t window_tokens  = resident_pages * kPagedKVPageSize;
        compact_tokens = main_tokens > shift ? main_tokens - shift : 0U;
        if (window_tokens != 0 && compact_tokens > window_tokens) { compact_tokens = window_tokens; }
        if (std::getenv("NINFER_WINDOW_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[win] main=%u window_base=%u shift=%u sink=%u compact=%u frontier=%u\n",
                         main_tokens, window_base, shift, sink_tokens, compact_tokens,
                         text_kv_addresses->committed_frontier(sequence.kv->text));
            std::fflush(stderr);
        }
    }
    if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-op] map main=%u backend=%u compact=%u active=%d\n", main_tokens,
                     backend_tokens, compact_tokens,
                     text_kv_addresses->active(sequence.kv->text) ? 1 : 0);
        std::fflush(stderr);
    }
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, compact_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, compact_backend,
                                                      device.stream);
    }
}

// Resident-window mode (Leg B): keep the newest absolute token range resident. Called at a chunk
// boundary, before new KV rows are appended, so the freed slots can be reused immediately.
void ProgramImpl::ensure_kv_window(SequenceState& sequence, std::uint32_t absolute_end) {
    if (!sequence.kv) { return; }
    static const std::uint32_t window_tokens = [] {
        const char* env = std::getenv("NINFER_RESIDENT_TOKENS");
        return env != nullptr && env[0] != '\0'
                   ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10))
                   : 0U;
    }();
    if (window_tokens == 0 || absolute_end <= window_tokens) { return; }
    // The sink pages are pinned beyond the moving window, so the tail gets what is left of the
    // resident pool: sink + tail must fit in the same page budget.
    const std::uint32_t sink_tokens = std::min<std::uint32_t>(
        kv_window_sink_tokens(),
        text_kv_addresses->entitlement(sequence.kv->text) * static_cast<std::uint32_t>(kPagedKVPageSize));
    const std::uint32_t tail_tokens =
        window_tokens > sink_tokens ? window_tokens - sink_tokens : window_tokens;
    // Round the base UP to a page boundary: the resident span must never exceed window_tokens.
    const std::uint32_t needed  = absolute_end - tail_tokens;
    const std::uint32_t desired_base =
        ((needed + kPagedKVPageSize - 1U) / kPagedKVPageSize) * kPagedKVPageSize;
    const std::uint32_t current_base = text_kv_addresses->window_base_tokens(sequence.kv->text);
    bool slid                        = false;
    if (desired_base > current_base) {
        const std::uint32_t sink_pages = std::min<std::uint32_t>(
            kv_window_sink_tokens() / kPagedKVPageSize,
            text_kv_addresses->entitlement(sequence.kv->text));
        const std::uint32_t drop_pages = (desired_base - current_base) / kPagedKVPageSize;
        if (std::getenv("NINFER_WINDOW_TRACE") != nullptr) {
            std::fprintf(stderr, "[win] slidepre base=%u desired=%u drop=%u sinkp=%u\n",
                         current_base, desired_base, drop_pages, sink_pages);
            std::fflush(stderr);
        }
        // Leg B / kvmem W3-a: archive the pages that are about to be released. Both coordinates are
        // needed: the pool is indexed by COMPACT logical page (keep_prefix .. keep_prefix + drop) and
        // the archive index is the ABSOLUTE token range. The pinned sink occupies the low slots and
        // is never dropped, so the moving tail starts at sink_pages * kPagedKVPageSize; window_base
        // accumulates exactly the tokens slid off that tail, so the first dropped slot holds
        // absolute token sink_pages * kPagedKVPageSize + current_base.
        if (KVArchive* archive = KVArchive::instance()) {
            // W3-b: slots [sink, sink + recall) are the recall destination, so once a swap has
            // landed there they hold a distant range and must not be indexed under this window's
            // coordinates. recall_into_tail archives the same slots under their real labels right
            // before it overwrites them, so no token is lost by this exclusion.
            const std::uint32_t skip_pages =
                g_kv_recall_armed ? std::min(kv_window_recall_pages(), drop_pages) : 0U;
            // The swap path re-indexes the slots it overwrites, so a block can already have been
            // written for part of what this slide is dropping. Everything strictly below the
            // high-water mark is indexed as a contiguous prefix, so dropping that leading run keeps
            // the index monotonic instead of appending a duplicate of the same tokens.
            std::uint32_t first_page = sink_pages + skip_pages;
            const std::uint32_t final_page = sink_pages + drop_pages;
            while (first_page < final_page &&
                   current_base + first_page * static_cast<std::uint32_t>(kPagedKVPageSize) <
                       g_kv_archive_watermark) {
                ++first_page;
            }
            const std::uint32_t first_absolute =
                current_base + first_page * static_cast<std::uint32_t>(kPagedKVPageSize);
            std::vector<DeviceKVPageHandle> dropped;
            dropped.reserve(drop_pages - skip_pages);
            for (std::uint32_t page = first_page; page < final_page; ++page) {
                dropped.push_back(text_kv_addresses->physical_page(sequence.kv->text, page));
            }
            if (!dropped.empty()) {
                if (std::getenv("NINFER_WINDOW_TRACE") != nullptr) {
                    std::fprintf(stderr,
                                 "[win] archive %zu pages from %u (first=%u final=%u base=%u "
                                 "watermark=%u)\n",
                                 dropped.size(), first_absolute, first_page, final_page,
                                 current_base, g_kv_archive_watermark);
                    std::fflush(stderr);
                }
                archive->archive(text_kv_pages->physical_pool(), dropped, first_absolute,
                                 device.stream);
                g_kv_archive_watermark =
                    current_base + final_page * static_cast<std::uint32_t>(kPagedKVPageSize);
            }
        }
        text_kv_addresses->slide_prefix_pages(
            sequence.kv->text, drop_pages, device.stream, sink_pages);
        // The speculative cache addresses the same tokens, so it has to slide with the main cache or
        // its compact frontier goes backwards and commit_frontier rejects it.
        if (sequence.kv->backend) {
            // The draft rows grow lazily, so they may still be narrower than the main row when a
            // slide arrives; widen them to the same span before dropping the prefix.
            const std::uint32_t backend_pages = std::min(
                text_kv_addresses->mapped_pages(sequence.kv->text),
                backend_kv_addresses->entitlement(*sequence.kv->backend));
            backend_kv_addresses->ensure_mapped_to_tokens(
                *sequence.kv->backend, backend_pages * static_cast<std::uint32_t>(kPagedKVPageSize),
                device.stream);
            backend_kv_addresses->slide_prefix_pages(
                *sequence.kv->backend, drop_pages, device.stream, sink_pages);
            // The draft rows slid with the main rows, so their compact coverage follows the same
            // base; mtp_kv_valid stays absolute and needs no correction.
        }
        slid = true;
        if (std::getenv("NINFER_WINDOW_TRACE") != nullptr) {
            std::fprintf(stderr, "[win] slide -> base=%u (abs_end=%u) frontier=%u\n",
                         text_kv_addresses->window_base_tokens(sequence.kv->text), absolute_end,
                         text_kv_addresses->committed_frontier(sequence.kv->text));
            std::fflush(stderr);
        }
    }
    const std::uint32_t base_now = text_kv_addresses->window_base_tokens(sequence.kv->text);
    const std::uint32_t sink_now = std::min(
        kv_window_sink_tokens(),
        text_kv_addresses->entitlement(sequence.kv->text) * static_cast<std::uint32_t>(kPagedKVPageSize));
    const std::uint32_t shift    = base_now == 0 ? 0U : base_now - sink_now;
    g_ninfer_kv_window_base_tokens = shift;
    // Re-grow the mapping to the window edge: a slide shrinks page_count, and the pool still owns
    // the freed pages, so the next chunk can materialize them again.
    text_kv_addresses->ensure_mapped_to_tokens(
        sequence.kv->text, absolute_end > shift ? absolute_end - shift : 0U, device.stream);
    // Retrieval mode waits for the decode step that follows the prompt: only then does the query
    // summary cover the question, and a ranking built from a mid-prefill chunk is meaningless.
    if (slid && KVRetrieval::instance() == nullptr) { recall_into_tail(sequence); }
    if (!slid && !g_kv_retrieval_done) {
        if (KVRetrieval* retrieval = KVRetrieval::instance();
            retrieval != nullptr && retrieval->prompt_complete()) {
            if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
                std::fprintf(stderr, "[kv-recall] trigger fired\n");
                std::fflush(stderr);
            }
            g_kv_retrieval_done = true;
            recall_into_tail(sequence);
        }
    }
}

// Leg B / kvmem W3-b: put an archived block back into the oldest tail slots.
//
// The slots right above the pinned sink hold the oldest resident tokens, which are exactly the
// pages the NEXT slide would drop. Swapping them for a distant archived block therefore costs only
// context that was already destined for eviction, while the attention mask stays satisfied: the
// kernel derives causality from the slot map, and identity ordering already reads "older than the
// query" for every slot below the frontier. The recalled K rows were rotated with their TRUE
// absolute positions at write time, so the scores are the full-context ones -- the slot index only
// decides visibility, not the rotation.
void ProgramImpl::recall_into_tail(SequenceState& sequence) {
    static thread_local std::uint32_t recall_calls = 0;
    static const std::uint32_t recall_first_token = [] {
        const char* env = std::getenv("NINFER_KV_RECALL_TOKEN");
        return env != nullptr ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10)) : 0U;
    }();
    static const std::uint32_t recall_slots = [] {
        const char* env = std::getenv("NINFER_KV_RECALL_PAGES");
        return env != nullptr ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10)) : 0U;
    }();
    KVArchive* archive = KVArchive::instance();
    if (archive == nullptr) { return; }
    KVRetrieval* retrieval = KVRetrieval::instance();
    if (recall_slots == 0 && retrieval == nullptr) { return; }
    const std::uint32_t sink_pages = std::min<std::uint32_t>(
        kv_window_sink_tokens() / kPagedKVPageSize,
        text_kv_addresses->entitlement(sequence.kv->text));
    const std::uint32_t mapped = text_kv_addresses->mapped_pages(sequence.kv->text);
    if (mapped <= sink_pages) { return; }
    // Default window is wide enough that a peak sitting a few pages early still covers the
    // document it points at; the backoff below keeps it symmetric around that peak.
    const std::uint32_t want = recall_slots != 0 ? recall_slots : 112U;
    // Never recall more pages than the archive actually holds: a window wider than the archived
    // set would overwrite live KV with pages that were never stored, which shows up as a window
    // that suddenly answers nonsense while every trace line still looks healthy.
    const std::uint32_t archived = static_cast<std::uint32_t>(archive->stats().pages);
    const std::uint32_t tail       = mapped > sink_pages ? mapped - sink_pages : 0U;
    const std::uint32_t reserve    = std::max<std::uint32_t>(16U, tail / 4U);
    const std::uint32_t capacity   = tail > reserve ? tail - reserve : 0U;
    // Never take more than half of the archive: a short conversation stores only a few blocks, and
    // recalling most of them turns the window into a copy of the document head.
    const std::uint32_t slots      = std::min({want, capacity, archived});
    if (slots == 0) { return; }
    // A window that is still wider than what has scrolled out does not need a recall: everything
    // is already resident, and writing recalled pages would trade live context for a copy of the
    // document head. Skip until the archive can actually supply a full block.
    if (archived < want) { return; }
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[kv-recall] attempt #%u mapped=%u sink=%u want=%u archived=%u slots=%u base=%u\n",
                     ++recall_calls,
                     mapped, sink_pages, want, archived, slots,
                     text_kv_addresses->window_base_tokens(sequence.kv->text));
        std::fflush(stderr);
    }
    // Evict-then-fill, and in this order: the destination slots hold real window tokens at this
    // instant (the slide just shifted the tail down into them), so index those bytes under their
    // true absolute positions BEFORE overwriting them. Without this the archive would freeze at the
    // first swap: once a slot holds recalled data the slide path has to skip it, so anything it
    // displaced here could never be reached again.
    //
    // The archiving has to happen BEFORE the recall sources are resolved: allocating can evict the
    // oldest archive blocks, and a view into a freed allocation is exactly what makes copy_from_host
    // fail with "Paged KV H2D geometry or extent is inconsistent". Resolving afterwards simply
    // returns nothing if the target block got evicted, which leaves the window untouched.
    const std::uint32_t base_now = text_kv_addresses->window_base_tokens(sequence.kv->text);
    const std::uint32_t dest_first =
        base_now + sink_pages * static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t page_bytes = static_cast<std::uint32_t>(kPagedKVPageSize);
    std::vector<DeviceKVPageHandle> evicted;
    std::uint32_t first_new = 0;
    for (; first_new < slots; ++first_new) {
        if (dest_first + first_new * page_bytes >= g_kv_archive_watermark) { break; }
    }
    evicted.reserve(slots - first_new);
    for (std::uint32_t index = first_new; index < slots; ++index) {
        evicted.push_back(text_kv_addresses->physical_page(sequence.kv->text, sink_pages + index));
    }
    if (!evicted.empty()) {
        archive->archive(text_kv_pages->physical_pool(), evicted,
                         dest_first + first_new * page_bytes, device.stream);
        g_kv_archive_watermark = dest_first + slots * page_bytes;
    }
    std::vector<KVArchive::RecallPage> sources;
    if (retrieval != nullptr) {
        // W4: rank every archived page against the current query instead of trusting a hand-picked
        // absolute token. Each block carries its own device mean-key record, one pointer per page.
        std::vector<std::uint16_t> means;
        std::vector<std::uint32_t> pages;
        const std::size_t mean_halves = static_cast<std::size_t>(kRetrievalPageMeanBytes) / 2U;
        // A page can be archived twice (a range that slid out, was recalled, then slid out again).
        // The newer copy may have been captured after its ring slot was recycled and therefore be
        // all zeros, so keep whichever copy actually carries a summary.
        std::unordered_map<std::uint32_t, std::size_t> slot_of_page;
        for (const KVArchive::Block& block : archive->blocks()) {
            if (block.means_host.size() < static_cast<std::size_t>(block.pages) * mean_halves) {
                continue;
            }
            for (std::uint32_t index = 0; index < block.pages; ++index) {
                const std::uint32_t page =
                    block.begin_token / static_cast<std::uint32_t>(kPagedKVPageSize) + index;
                const std::uint16_t* source = block.means_host.data() + index * mean_halves;
                const auto found = slot_of_page.find(page);
                if (found == slot_of_page.end()) {
                    slot_of_page.emplace(page, pages.size());
                    pages.push_back(page);
                    means.insert(means.end(), source, source + mean_halves);
                } else if (means[found->second * mean_halves] == 0U && source[0] != 0U) {
                    std::copy(source, source + mean_halves,
                              means.begin() + static_cast<std::ptrdiff_t>(found->second * mean_halves));
                }
            }
        }
        if (std::getenv("NINFER_KV_PROBE_PAGE") != nullptr) {
            for (const KVArchive::Block& block : archive->blocks()) {
                const std::uint16_t sample =
                    block.means_host.size() > 2U ? block.means_host[1] : 0xFFFFU;
                std::fprintf(stderr, "[kv-blk] begin=%u pages=%u halves=%zu sample=%u buf=%p\n",
                             block.begin_token, block.pages, block.means_host.size(),
                             static_cast<unsigned>(sample),
                             static_cast<const void*>(block.means_host.data()));
            }
            std::fflush(stderr);
        }
        const auto ranked =
            retrieval->rank(means.empty() ? nullptr : static_cast<const void*>(means.data()), pages,
                            device.stream);
        if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE"); probe_env != nullptr) {
            const std::uint32_t probe = static_cast<std::uint32_t>(std::atoi(probe_env));
            for (std::size_t index = 0; index < pages.size(); ++index) {
                if (pages[index] != probe) { continue; }
                const std::uint16_t* entry = means.data() + index * mean_halves;
                std::fprintf(stderr, "[kv-host] page=%u index=%zu h0=%u h1=%u h2=%u\n", probe, index,
                             static_cast<unsigned>(entry[0]), static_cast<unsigned>(entry[1]),
                             static_cast<unsigned>(entry[2]));
            }
            std::fflush(stderr);
        }
        // Two-stage rerank: the mean summaries are only a coarse filter, so the best candidates are
        // re-scored against their decoded int4 keys before the swap-in choice is made.
        std::vector<std::decay_t<decltype(ranked[0])>> rescored(ranked.begin(), ranked.end());
        const char* rerank_env = std::getenv("NINFER_KV_RERANK");
        const bool rerank_on = rerank_env == nullptr || rerank_env[0] != '0';
        if (rerank_on && rescored.size() > 1U) {
            std::vector<float> query_rows(static_cast<std::size_t>(kRetrievalLayers) *
                                              kRetrievalKVHeads * 256U,
                                          0.0F);
            retrieval->export_query(query_rows.data(), device.stream);
            std::unordered_map<std::uint32_t, const void*> bytes_of_page;
            for (const auto& source : sources) {
                bytes_of_page.emplace(source.absolute_token / kPagedKVPageSize, source.source.data());
            }
            const std::size_t limit = std::min<std::size_t>(rescored.size(), 256U);
            for (std::size_t index = 0; index < limit; ++index) {
                // The sink pages hold the prompt template and the task statement, which match the
                // query strongly without containing the answer; keep them out of the rerank.
                if (rescored[index].absolute_page < 32U) {
                    rescored[index].score = -1.0e30F;
                    continue;
                }
                const auto found = bytes_of_page.find(rescored[index].absolute_page);
                if (found != bytes_of_page.end()) {
                    rescored[index].score =
                        KVRetrieval::score_page_keys(found->second, query_rows.data());
                }
            }
            std::stable_sort(rescored.begin(), rescored.begin() + static_cast<std::ptrdiff_t>(limit),
                             [](const auto& lhs, const auto& rhs) { return lhs.score > rhs.score; });
            if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
                std::fprintf(stderr, "[kv-refine] top3:");
                for (std::size_t index = 0; index < limit && index < 3U; ++index) {
                    std::fprintf(stderr, " %u@%.5f", rescored[index].absolute_page,
                                 static_cast<double>(rescored[index].score));
                }
                if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE");
                    probe_env != nullptr) {
                    const std::uint32_t probe = static_cast<std::uint32_t>(std::atoi(probe_env));
                    for (std::size_t index = 0; index < limit; ++index) {
                        if (rescored[index].absolute_page != probe) { continue; }
                        std::fprintf(stderr, " | probe=%u rank=%zu score=%.5f", probe, index,
                                     static_cast<double>(rescored[index].score));
                        break;
                    }
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
        }
        if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
            const KVArchiveStats archive_stats = archive->stats();
            std::fprintf(stderr, "[kv-rank] blocks=%llu pages=%llu evicted=%llu candidates=%zu top:",
                         static_cast<unsigned long long>(archive_stats.blocks),
                         static_cast<unsigned long long>(archive_stats.pages),
                         static_cast<unsigned long long>(archive_stats.evicted), ranked.size());            for (std::size_t index = 0; index < ranked.size() && index < 5; ++index) {
                std::fprintf(stderr, " %u@%.6f", ranked[index].absolute_page,
                             static_cast<double>(ranked[index].score));
            }
            std::fprintf(stderr, "\n");
            std::uint32_t held = 0;
            std::uint32_t lowest = 0xFFFFFFFFU;
            std::uint32_t highest = 0;
            for (std::size_t index = 0; index < ranked.size() && index < slots; ++index) {
                const std::uint32_t page = ranked[index].absolute_page;
                lowest  = std::min(lowest, page);
                highest = std::max(highest, page);
                if (page >= 768 && page < 800) { ++held; }
            }
            std::fprintf(stderr, "[kv-rank] selected %zu pages range [%u,%u] needle_hits=%u\n",
                         ranked.size() < slots ? ranked.size() : slots, lowest, highest, held);
            std::fprintf(stderr, "[kv-rank] needle zone best:");
            int shown = 0;
            for (std::size_t index = 0; index < ranked.size() && shown < 3; ++index) {
                if (ranked[index].absolute_page >= 768 && ranked[index].absolute_page < 800) {
                    std::fprintf(stderr, " #%zu %u@%.6f", index, ranked[index].absolute_page,
                                 static_cast<double>(ranked[index].score));
                    ++shown;
                }
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        // The ranker peaks near the document that holds the answer rather than exactly on it, so
        // each hit expands forward into its neighbours before the next hit is considered.
        if (!ranked.empty()) {
            const char* div_env = std::getenv("NINFER_KV_RECALL_DIV");
            const std::uint32_t divisor =
                div_env != nullptr ? std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(
                                                                         std::strtoul(div_env, nullptr, 10)))
                                   : (archived < 8U * static_cast<std::uint32_t>(slots) ? 4U : 2U);
            const std::uint32_t block =
                std::max<std::uint32_t>(16U, static_cast<std::uint32_t>(slots) / divisor);
            std::vector<std::uint32_t> centres;
            std::vector<std::uint32_t> taken;
            for (const auto& candidate : rescored) {
                if (sources.size() >= slots) { break; }
                const std::uint32_t page = candidate.absolute_page;
                bool already = false;
                for (const std::uint32_t done : taken) {
                    if (done == page) { already = true; break; }
                }
                if (already) { continue; }
                const std::uint32_t left = static_cast<std::uint32_t>(slots) -
                                           static_cast<std::uint32_t>(sources.size());
                const std::uint32_t width = std::max<std::uint32_t>(16U, std::min(block, left));
                // Centre the block on the peak: a peak that lands a few pages past the needle would
                // otherwise start the block at the needle and run away from it.
                const std::uint32_t back  = block / 2U;
                const std::uint32_t begin = page > back ? page - back : 0U;
                const std::vector<KVArchive::RecallPage> run =
                    archive->recall_pages(begin * kPagedKVPageSize, width);
                if (run.empty()) { continue; }
                centres.push_back(page);
                sources.insert(sources.end(), run.begin(), run.end());
                for (std::uint32_t offset = 0; offset < run.size(); ++offset) {
                    taken.push_back(begin + offset);
                }
            }
            // A small window leaves budget after the wide peaks; spend it on the next ranked pages so a
            // second peak sitting outside every block still gets pulled in.
            const char* fill_env = std::getenv("NINFER_KV_RECALL_FILL");
            const bool fill_extra = fill_env == nullptr || fill_env[0] != '0';
            for (const auto& candidate : rescored) {
                if (!fill_extra || sources.size() >= slots) { break; }
                const std::uint32_t page = candidate.absolute_page;
                bool covered = false;
                for (const std::uint32_t done : taken) {
                    if (done == page) { covered = true; break; }
                }
                if (covered) { continue; }
                const std::vector<KVArchive::RecallPage> run =
                    archive->recall_pages(page * kPagedKVPageSize, 1);
                if (run.empty()) { continue; }
                taken.push_back(page);
                sources.insert(sources.end(), run.begin(), run.end());
            }
            if (sources.size() > slots) { sources.resize(slots); }
            if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
                std::fprintf(stderr, "[kv-recall] pages:");
                for (std::size_t index = 0; index < sources.size() && index < 24; ++index) {
                    std::fprintf(stderr, " %u", sources[index].absolute_token / kPagedKVPageSize);
                }
                std::fprintf(stderr, " ... tail:");
                for (std::size_t index = sources.size() > 8 ? sources.size() - 8 : 0; index < sources.size(); ++index) {
                    std::fprintf(stderr, " %u", sources[index].absolute_token / kPagedKVPageSize);
                }
                std::fprintf(stderr, " (of %zu)\n", sources.size());
                std::fflush(stderr);
            }
        }
    } else {
        sources = archive->recall_pages(recall_first_token, slots);
    }
    if (sources.empty()) { return; }
    if (std::getenv("NINFER_KV_DEQ_TRACE") != nullptr) {
        for (std::size_t index = 0; index < sources.size() && index < 2U; ++index) {
            KVRetrieval::debug_scan_page_keys(sources[index].source.data(),
                                              static_cast<std::uint32_t>(sources[index].absolute_token /
                                                                         kPagedKVPageSize));
        }
    }
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const DeviceKVPageHandle destination =
            text_kv_addresses->physical_page(sequence.kv->text, sink_pages + index);
        text_kv_pages->physical_pool().copy_from_host(
            sources[index].source, std::span<const DeviceKVPageHandle>(&destination, 1),
            device.stream);
    }
    g_kv_recall_armed  = true;
    g_kv_recall_loaded = true;
    if (std::getenv("NINFER_KV_ARCHIVE_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[kv-recall] %zu pages from token %u -> slots [%u,%zu) (mapped %u)\n",
                     sources.size(), sources.front().absolute_token, sink_pages,
                     static_cast<std::size_t>(sink_pages) + sources.size(), mapped);
        std::fflush(stderr);
    }
}

void ProgramImpl::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                     std::uint32_t backend_tokens) {
    if (std::getenv("NINFER_KV_OP_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-op] commit main=%u backend=%u\n", main_tokens, backend_tokens);
        std::fflush(stderr);
    }
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    // Resident-window mode: the address space is addressed in compact coordinates, so the committed
    // frontier must be expressed relative to the window base (absolute positions stay untouched).
    std::uint32_t main_frontier    = main_tokens;
    std::uint32_t backend_frontier = backend_tokens;
    static const bool resident_windowed = [] {
        const char* env = std::getenv("NINFER_RESIDENT_TOKENS");
        return env != nullptr && env[0] != '\0';
    }();
    if (resident_windowed) {
        // Decode appends one token per step, so the window has to advance here as well: prefill
        // only slides at chunk boundaries, which never happen while generating.
        ensure_kv_window(sequence, main_tokens);
        main_frontier = main_tokens > g_ninfer_kv_window_base_tokens
                            ? main_tokens - g_ninfer_kv_window_base_tokens
                            : 0U;
        if (sequence.kv->backend && backend_tokens != 0) {
            backend_frontier = backend_tokens > g_ninfer_kv_window_base_tokens
                                   ? backend_tokens - g_ninfer_kv_window_base_tokens
                                   : 0U;
            const std::uint32_t backend_page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t backend_capacity =
                std::min(backend_kv_addresses->entitlement(*sequence.kv->backend),
                         backend_kv_addresses->mapped_pages(*sequence.kv->backend)) *
                backend_page_size;
            if (backend_frontier > backend_capacity) { backend_frontier = backend_capacity; }
        }
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_frontier);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_frontier);
    }
}

void ProgramImpl::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    // Resident-window mode: destructive truncation also speaks compact coordinates.
    std::uint32_t main_trim = main_tokens;
    static const bool windowed_trim = [] {
        const char* env = std::getenv("NINFER_RESIDENT_TOKENS");
        return env != nullptr && env[0] != '\0';
    }();
    if (windowed_trim) {
        const std::uint32_t window_base = text_kv_addresses->window_base_tokens(sequence.kv->text);
        const std::uint32_t sink_now =
            std::min(kv_window_sink_tokens(),
                     text_kv_addresses->entitlement(sequence.kv->text) *
                         static_cast<std::uint32_t>(kPagedKVPageSize));
        const std::uint32_t shift = window_base == 0 ? 0U : window_base - sink_now;
        g_ninfer_kv_window_base_tokens = shift;
        main_trim = main_tokens > shift ? main_tokens - shift : 0U;
        if (std::getenv("NINFER_WINDOW_TRACE") != nullptr) {
            std::fprintf(stderr, "[win] trim main=%u window_base=%u shift=%u -> compact=%u\n",
                         main_tokens, window_base, shift, main_trim);
            std::fflush(stderr);
        }
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_trim);
    if (sequence.kv->backend) {
        // Same compact coordinates as the commit above: the speculative cache is window addressed.
        std::uint32_t backend_trim =
            backend_tokens > g_ninfer_kv_window_base_tokens
                ? backend_tokens - g_ninfer_kv_window_base_tokens
                : 0U;
        const std::uint32_t backend_committed =
            backend_kv_addresses->committed_frontier(*sequence.kv->backend);
        if (backend_trim > backend_committed) { backend_trim = backend_committed; }
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_trim);
    }
}

void ProgramImpl::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        text_kv_addresses->release_growth_entitlement(sequence.kv->text);
        if (sequence.kv->backend) {
            backend_kv_addresses->release_growth_entitlement(*sequence.kv->backend);
        }
    } catch (...) {}
}

void ProgramImpl::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (std::getenv("NINFER_KV_RELEASE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-release] active seq=%p pages=%u frontier=%u\n",
                     static_cast<const void*>(&sequence),
                     text_kv_addresses->entitlement(sequence.kv->text),
                     text_kv_addresses->committed_frontier(sequence.kv->text));
        std::fflush(stderr);
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses || !text_kv_addresses->can_release(sequence.kv->text)) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        (!backend_kv_addresses || !backend_kv_addresses->can_release(*sequence.kv->backend))) {
        std::terminate();
    }
    if (std::getenv("NINFER_KV_RELEASE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-release] strict seq=%p pages=%u frontier=%u\n",
                     static_cast<const void*>(&sequence),
                     text_kv_addresses->entitlement(sequence.kv->text),
                     text_kv_addresses->committed_frontier(sequence.kv->text));
        std::fflush(stderr);
    }
    if (sequence.kv->backend && !backend_kv_addresses->release(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    if (std::getenv("NINFER_KV_RELEASE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-release] plain seq=%p pages=%u frontier=%u\n",
                     static_cast<const void*>(&sequence),
                     text_kv_addresses->entitlement(sequence.kv->text),
                     text_kv_addresses->committed_frontier(sequence.kv->text));
        std::fflush(stderr);
    }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

qwen3_5::PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

qwen3_5::PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImpl::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}


} // namespace ninfer::models::qwen3_5::detail
