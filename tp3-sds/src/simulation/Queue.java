package simulation;

import java.util.List;

/**
 * Models the physical and logical state of a waiting line.
 *
 * <p>Index 0 is the front (closest to the server). Spots have fixed physical
 * {@link Position}s defined by the concrete implementation. This queue stores
 * only the clients that are physically occupying queue spots. Clients that are
 * still approaching the queue are tracked by the simulation layer and only call
 * {@link #enqueue} once they actually reach the current take-position.
 */
public interface Queue {

    /** Maximum number of spots this queue can hold. */
    int capacity();

    /** Number of clients currently assigned to this queue. */
    int size();

    /** Physical position of the spot at the given logical index. */
    Position getSpotPosition(int index);

    /**
     * Adds a client to the back of the physically occupied queue and assigns its
     * {@code queueSpotIndex}.
     *
     * @return the spot index that was assigned (= size - 1 after insertion)
     */
    int enqueue(Client client);

    /**
     * Removes the front client (index 0) and shifts all remaining clients'
     * {@code queueSpotIndex} down by 1 (logical shift only — physical movement
     * is handled via advance events).
     *
     * @return list of advancements needed so the caller can schedule advance events
     */
    List<QueueAdvancement> dequeue();

    /** Returns the client at the given logical index, or {@code null} if out of range. */
    Client getClientAt(int index);

    /** Ordered snapshot of all clients currently in the queue (index 0 = front). */
    List<Client> getClients();

    /** Identifier of the server this queue belongs to (-1 for a shared queue). */
    int getOwnerId();
}
