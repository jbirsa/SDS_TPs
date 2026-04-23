package simulation;

/**
 * A client finishes walking toward the current queue take-position.
 */
public class ClientArrivesAtQueueSpotEvent extends Event {

    public final int clientId;
    public final int movementVersion;

    public ClientArrivesAtQueueSpotEvent(double time, int clientId, int movementVersion) {
        super(time);
        this.clientId = clientId;
        this.movementVersion = movementVersion;
    }
}
