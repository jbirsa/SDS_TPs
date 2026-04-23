package simulation;

import java.awt.Color;

public class Client {
    public final int id;
    public final double arrivalTime;
    public final int[] rgb; // [r, g, b] derived from id via golden-ratio hue

    private Position position;
    private ClientState state;
    private int assignedServerId; // -1 if unassigned (modality B before reaching server)
    private int queueSpotIndex;   // logical index in queue, 0 = front; -1 if not in queue
    private Position movementStart;
    private Position movementTarget;
    private double movementStartTime;
    private double movementEndTime;
    private int movementVersion;

    public Client(int id, Position spawnPosition, double arrivalTime) {
        this.id = id;
        this.arrivalTime = arrivalTime;
        this.position = spawnPosition;
        this.state = ClientState.WALKING_TO_QUEUE_SPOT;
        this.assignedServerId = -1;
        this.queueSpotIndex = -1;
        this.movementStart = null;
        this.movementTarget = null;
        this.movementStartTime = 0.0;
        this.movementEndTime = 0.0;
        this.movementVersion = 0;
        this.rgb = deriveColorFromId(id);
    }

    /** Spreads client colors evenly across the hue wheel using the golden ratio. */
    private static int[] deriveColorFromId(int id) {
        float hue = (float) ((id * 0.618033988749895) % 1.0);
        Color c = Color.getHSBColor(hue, 0.85f, 0.90f);
        return new int[]{c.getRed(), c.getGreen(), c.getBlue()};
    }

    public Position getPosition()              { return position; }
    public void setPosition(Position p)        { this.position = p; }
    public ClientState getState()              { return state; }
    public void setState(ClientState s)        { this.state = s; }
    public int getAssignedServerId()           { return assignedServerId; }
    public void setAssignedServerId(int sid)   { this.assignedServerId = sid; }
    public int getQueueSpotIndex()             { return queueSpotIndex; }
    public void setQueueSpotIndex(int idx)     { this.queueSpotIndex = idx; }

    public boolean hasActiveMovement() {
        return movementStart != null && movementTarget != null;
    }

    public int startMovement(Position target, double startTime, double endTime) {
        this.movementVersion++;
        this.movementStart = this.position;
        this.movementTarget = target;
        this.movementStartTime = startTime;
        this.movementEndTime = endTime;
        if (endTime <= startTime) {
            this.position = target;
        }
        return movementVersion;
    }

    public void updatePositionAt(double time) {
        if (!hasActiveMovement()) return;
        if (movementEndTime <= movementStartTime || time >= movementEndTime) {
            this.position = movementTarget;
            return;
        }
        if (time <= movementStartTime) {
            this.position = movementStart;
            return;
        }

        double alpha = (time - movementStartTime) / (movementEndTime - movementStartTime);
        double x = movementStart.x + alpha * (movementTarget.x - movementStart.x);
        double y = movementStart.y + alpha * (movementTarget.y - movementStart.y);
        this.position = new Position(x, y);
    }

    public void finishMovement() {
        if (movementTarget != null) {
            this.position = movementTarget;
        }
        clearMovement();
    }

    public void clearMovement() {
        this.movementStart = null;
        this.movementTarget = null;
        this.movementStartTime = 0.0;
        this.movementEndTime = 0.0;
    }

    public int getMovementVersion() {
        return movementVersion;
    }
}
