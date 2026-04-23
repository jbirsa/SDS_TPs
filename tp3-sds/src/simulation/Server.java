package simulation;

/**
 * Represents one server on the south wall of the room.
 *
 * <p>A server is in one of three availability states:
 * <ol>
 *   <li>Free — no client assigned.
 *   <li>Reserved — a client has been dispatched and is walking toward the server
 *       (service has not yet started). The server must NOT accept another client
 *       during this window.
 *   <li>Busy — a client is being served.
 * </ol>
 */
public class Server {
    public final int      id;
    public final Position position;

    private boolean busy;
    private int     pendingClientId;  // id of client walking toward this server (-1 = none)
    private Client  currentClient;

    public Server(int id, Position position) {
        this.id              = id;
        this.position        = position;
        this.busy            = false;
        this.pendingClientId = -1;
        this.currentClient   = null;
    }

    /** {@code true} only when neither busy nor reserved. */
    public boolean isAvailable() {
        return !busy && pendingClientId == -1;
    }

    public boolean isBusy() { return busy; }

    public Client getCurrentClient() { return currentClient; }

    /**
     * Reserves this server for a client that is currently walking toward it.
     * After calling this the server is no longer available for other clients.
     */
    public void reserveFor(int clientId) {
        this.pendingClientId = clientId;
    }

    /**
     * Called when the reserved/pending client physically arrives and service begins.
     */
    public void startServing(Client client) {
        this.busy            = true;
        this.pendingClientId = -1;
        this.currentClient   = client;
        client.setState(ClientState.BEING_SERVED);
        client.setPosition(this.position);
    }

    /**
     * Called when service ends. Returns the client who was served and resets
     * the server to the Free state.
     */
    public Client finishServing() {
        Client served    = this.currentClient;
        this.busy        = false;
        this.currentClient = null;
        return served;
    }
}
