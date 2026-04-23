package simulation;

/**
 * A new client spawns at a random position inside the room.
 * Processed by {@link Simulation#processClientArrival}.
 */
public class ClientArrivalEvent extends Event {

    public ClientArrivalEvent(double time) {
        super(time);
    }
}
