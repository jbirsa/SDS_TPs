package simulation;

/**
 * A client (previously at queue front) finishes walking to the server and service begins.
 */
public class ClientArrivesAtServerEvent extends Event {

    public final int clientId;
    public final int serverId;
    public final int movementVersion;

    public ClientArrivesAtServerEvent(double time, int clientId, int serverId, int movementVersion) {
        super(time);
        this.clientId = clientId;
        this.serverId = serverId;
        this.movementVersion = movementVersion;
    }
}
