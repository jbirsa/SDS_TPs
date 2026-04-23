package simulation;

/**
 * Carries the data needed to schedule one queue-advance event:
 * the client moving from oldSpotIndex to newSpotIndex (= oldSpotIndex - 1).
 */
public class QueueAdvancement {
    public final Client client;
    public final int oldSpotIndex;
    public final int newSpotIndex;

    public QueueAdvancement(Client client, int oldSpotIndex, int newSpotIndex) {
        this.client = client;
        this.oldSpotIndex = oldSpotIndex;
        this.newSpotIndex = newSpotIndex;
    }
}
