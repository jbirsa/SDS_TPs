package simulation;

/**
 * A queued client finishes moving one spot forward (from index {@code newSpotIndex + 1}
 * to {@code newSpotIndex}) after a service completion triggered the chain advance.
 */
public class ClientAdvancesInQueueEvent extends Event {

    public final int clientId;
    public final int newSpotIndex;
    public final int movementVersion;

    public ClientAdvancesInQueueEvent(double time, int clientId, int newSpotIndex, int movementVersion) {
        super(time);
        this.clientId     = clientId;
        this.newSpotIndex = newSpotIndex;
        this.movementVersion = movementVersion;
    }
}
