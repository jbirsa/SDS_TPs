package simulation;

/**
 * A server finishes serving the current client. The client leaves the system,
 * the server becomes free, and the queue advance chain is triggered.
 */
public class ServiceCompletionEvent extends Event {

    public final int serverId;
    public final int clientId; // the client being served (for cross-checking)

    public ServiceCompletionEvent(double time, int serverId, int clientId) {
        super(time);
        this.serverId = serverId;
        this.clientId = clientId;
    }
}
