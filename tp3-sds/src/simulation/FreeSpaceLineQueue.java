package simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Queue whose spots are generated incrementally in free space.
 *
 * <p>The first client stands directly in front of the server/anchor. Every new
 * spot is placed 1 m away from the previous spot by rotating the previous
 * segment with a random angle, while keeping the queue inside the room.
 */
public class FreeSpaceLineQueue implements Queue {

    private static final int    MAX_CAPACITY = 20000;
    private static final double SPOT_SPACING = 1.0;
    private static final double MIN_CLEARANCE = 0.85;
    private static final int    MAX_TRIES = 32;

    private final int            ownerId;
    private final Position       anchorPosition;
    private final double         minX;
    private final double         maxX;
    private final double         maxY;
    private final Random         random;
    private final List<Position> spotPositions;
    private final List<Client>   clients;   // index 0 = front

    public FreeSpaceLineQueue(int ownerId,
                              Position anchorPosition,
                              double stripWidth,
                              double roomSize,
                              Random random) {
        this.ownerId        = ownerId;
        this.anchorPosition = anchorPosition;
        this.minX           = Math.max(0.5, anchorPosition.x - stripWidth / 2.0);
        this.maxX           = Math.min(roomSize - 0.5, anchorPosition.x + stripWidth / 2.0);
        this.maxY           = roomSize - 0.5;
        this.random         = random;
        this.spotPositions  = new ArrayList<>();
        this.clients        = new ArrayList<>();
    }

    @Override
    public Position getSpotPosition(int index) {
        ensureSpotExists(index);
        return spotPositions.get(index);
    }

    @Override public int capacity()            { return MAX_CAPACITY; }
    @Override public int size()                { return clients.size(); }
    @Override public int getOwnerId()          { return ownerId; }
    @Override public List<Client> getClients() { return new ArrayList<>(clients); }

    @Override
    public Client getClientAt(int index) {
        if (index < 0 || index >= clients.size()) return null;
        return clients.get(index);
    }

    @Override
    public int enqueue(Client client) {
        int spotIndex = clients.size();
        ensureSpotExists(spotIndex);
        clients.add(client);
        client.setQueueSpotIndex(spotIndex);
        return spotIndex;
    }

    @Override
    public List<QueueAdvancement> dequeue() {
        List<QueueAdvancement> advancements = new ArrayList<>();
        if (clients.isEmpty()) return advancements;

        clients.remove(0);

        for (int newIdx = 0; newIdx < clients.size(); newIdx++) {
            Client c      = clients.get(newIdx);
            int    oldIdx = newIdx + 1;
            advancements.add(new QueueAdvancement(c, oldIdx, newIdx));
            c.setQueueSpotIndex(newIdx);
        }

        // Drop the last cached spot so the next arriving client gets a freshly
        // generated position with a new random angle, independent of any previous
        // particle that occupied that queue position.
        while (spotPositions.size() > clients.size()) {
            spotPositions.remove(spotPositions.size() - 1);
        }

        return advancements;
    }

    private void ensureSpotExists(int index) {
        if (index >= MAX_CAPACITY) {
            throw new IllegalStateException("Queue capacity exceeded: " + MAX_CAPACITY);
        }

        while (spotPositions.size() <= index) {
            if (spotPositions.isEmpty()) {
                spotPositions.add(firstSpot());
            } else {
                spotPositions.add(nextSpot());
            }
        }
    }

    private Position firstSpot() {
        return new Position(anchorPosition.x, anchorPosition.y + SPOT_SPACING);
    }

    private Position nextSpot() {
        Position previous = spotPositions.get(spotPositions.size() - 1);

        // Pick a fully independent random angle in the upper semicircle [0, π].
        // This angle is NOT derived from the previous segment direction, so every
        // new spot is placed at a fresh random direction from the current tail.
        for (int attempt = 0; attempt < MAX_TRIES; attempt++) {
            double angle = random.nextDouble() * Math.PI; // [0, π] → sin ≥ 0 (upward)
            double dx = Math.cos(angle);
            double dy = Math.sin(angle);
            Position candidate = advance(previous, dx, dy);
            if (isValid(previous, candidate)) {
                return candidate;
            }
        }

        // Fallback: straight up (allow growing past the room's physical height
        // for statistics-only, unstable runs).
        Position straightUp = new Position(previous.x, previous.y + SPOT_SPACING);
        if (isValid(previous, straightUp)) {
            return straightUp;
        }

        Position fallback = new Position(clamp(previous.x, minX, maxX),
                                         previous.y + SPOT_SPACING);
        if (isValid(previous, fallback)) {
            return fallback;
        }

        // Last-resort: accept the fallback even if clearance is tight (stats-only).
        return fallback;
    }

    private Position advance(Position origin, double dx, double dy) {
        double norm = Math.hypot(dx, dy);
        if (norm == 0.0) {
            dx = 0.0;
            dy = 1.0;
            norm = 1.0;
        }
        return new Position(origin.x + SPOT_SPACING * dx / norm,
                            origin.y + SPOT_SPACING * dy / norm);
    }

    private boolean isValid(Position previous, Position candidate) {
        if (candidate.x < minX || candidate.x > maxX) return false;
        if (candidate.y < 0.5) return false;
        // Allow y to grow past the physical room height: for unstable, long-running
        // simulations the queue must accept many clients; their exact off-room
        // position only affects travel time, not the stability analysis.
        if (candidate.distanceTo(anchorPosition) <= previous.distanceTo(anchorPosition)) return false;

        for (Position spot : spotPositions) {
            if (candidate.distanceTo(spot) < MIN_CLEARANCE) {
                return false;
            }
        }
        return true;
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
