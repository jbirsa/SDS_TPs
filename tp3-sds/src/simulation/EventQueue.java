package simulation;

import java.util.PriorityQueue;

/**
 * Min-heap of {@link Event}s ordered by scheduled time. Wraps
 * {@link PriorityQueue} and exposes only the methods the simulation needs.
 */
public class EventQueue {

    private final PriorityQueue<Event> heap = new PriorityQueue<>();

    public void add(Event event) {
        heap.add(event);
    }

    /** Returns and removes the next event, or {@code null} if empty. */
    public Event poll() {
        return heap.poll();
    }

    public boolean isEmpty() {
        return heap.isEmpty();
    }

    public int size() {
        return heap.size();
    }
}
