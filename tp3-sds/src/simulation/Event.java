package simulation;

/**
 * Base class for all discrete events in the simulation.
 *
 * <p>Events are ordered by their scheduled {@code time}; ties are broken by
 * insertion order (via a secondary counter) to keep the simulation deterministic.
 */
public abstract class Event implements Comparable<Event> {

    private static long counter = 0;

    public final double time;
    private final long  insertionOrder;

    protected Event(double time) {
        this.time           = time;
        this.insertionOrder = counter++;
    }

    @Override
    public int compareTo(Event other) {
        int cmp = Double.compare(this.time, other.time);
        if (cmp != 0) return cmp;
        return Long.compare(this.insertionOrder, other.insertionOrder);
    }
}
