#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

// Forward declarations for Vulkan types to avoid leaking headers
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDeviceMemory_T* VkDeviceMemory;
typedef struct VkFence_T* VkFence;
typedef uint64_t VkDeviceSize;

namespace mm_rec {

/**
 * Interface for Compute Backend (GPU/CPU)
 * Abstracts hardware initialization and resource management.
 */
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    /**
     * Initialize the backend.
     * @return true if successful
     */
    virtual bool init() = 0;

    /**
     * Check if backend is ready.
     */
    virtual bool is_ready() const = 0;

    /**
     * Create a storage buffer.
     * @param size Size in bytes
     * @param buffer Output buffer handle
     * @param memory Output memory handle
     * @return true if successful
     */
    virtual bool create_buffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory) = 0;

    /**
     * Free memory.
     */
    virtual void free_memory(VkDeviceMemory memory) = 0;

    /**
     * Create a fence for synchronization.
     */
    virtual VkFence create_fence() = 0;

    /**
     * Wait for a fence.
     */
    virtual void wait_fence(VkFence fence, uint64_t timeout = (~0ULL)) = 0;

    /**
     * Reset a fence.
     */
    virtual void reset_fence(VkFence fence) = 0;

    /**
     * Destroy a fence.
     */
    virtual void destroy_fence(VkFence fence) = 0;

    /**
     * Set VRAM reservation.
     */
    virtual void set_reservation(size_t mb) = 0;

    /**
     * Get device name.
     */
    virtual std::string get_device_name() const = 0;

    /**
     * Get total VRAM in bytes.
     */
    virtual size_t get_total_vram() const = 0;
};

} // namespace mm_rec
