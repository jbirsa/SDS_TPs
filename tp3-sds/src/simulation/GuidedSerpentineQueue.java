package simulation;

import java.util.ArrayList;
import java.util.List;

/**
 * Queue whose spots form a serpentine (snake) pattern starting directly above
 * the server and filling the server's horizontal strip row by row.
 *
 * <p>Layout (strip width W = 30 / k, server at (sx, 0)):
 * <ul>
 *   <li>Row y = 1 : from sx leftward to leftEdge of strip
 *   <li>Row y = 2 : from leftEdge rightward to rightEdge
 *   <li>Row y = 3 : from rightEdge leftward …
 * </ul>
 * Spots are 1 m apart. The queue grows as far into the room as needed up to
 * {@code MAX_CAPACITY} spots.
 */
public class GuidedSerpentineQueue implements Queue {

    /**
     * Soft cap on spot layout. The physically meaningful serpentine fills only
     * up to this many positions; past that, spots are extended virtually (still
     * at 1 m spacing) so statistics on unstable, long-running simulations do
     * not crash with OOB. Pick a number comfortably larger than the worst-case
     * expected queue length.
     */
    private static final int MAX_CAPACITY = 20000;
    private static final double ROOM_MAX  = 29.5;

    private final int ownerId;
    private final List<Position> spotPositions;
    private final List<Client>   clients;       // index 0 = front

    public GuidedSerpentineQueue(int ownerId, Position serverPosition, double stripWidth) {
        this.ownerId       = ownerId;
        this.spotPositions = buildSerpentine(serverPosition, stripWidth);
        this.clients       = new ArrayList<>();
    }

    private List<Position> buildSerpentine(Position serverPos, double stripWidth) {
        List<Position> spots = new ArrayList<>();

        double sx       = serverPos.x;
        double leftX    = Math.max(0.5, sx - stripWidth / 2.0);
        double rightX   = Math.min(ROOM_MAX, sx + stripWidth / 2.0);

        int sxInt       = (int) Math.round(sx);
        int leftXInt    = (int) Math.ceil(leftX);
        int rightXInt   = (int) Math.floor(rightX);

        // Leave one empty column next to internal strip boundaries so adjacent
        // serpentine queues do not overlap on the shared division line.
        if (leftX > 0.5) {
            leftXInt += 1;
        }
        if (rightX < ROOM_MAX) {
            rightXInt -= 1;
        }

        double baseY = serverPos.y + 1.0;

        // First row at baseY: from sx going left to leftX (starts above anchor)
        for (int xi = sxInt; xi >= leftXInt && spots.size() < MAX_CAPACITY; xi--) {
            spots.add(new Position(xi, baseY));
        }

        // Subsequent rows: alternating direction across full strip width
        boolean goRight = true;
        double y = baseY + 1.0;
        while (spots.size() < MAX_CAPACITY) {
            if (goRight) {
                for (int xi = leftXInt; xi <= rightXInt && spots.size() < MAX_CAPACITY; xi++) {
                    spots.add(new Position(xi, y));
                }
            } else {
                for (int xi = rightXInt; xi >= leftXInt && spots.size() < MAX_CAPACITY; xi--) {
                    spots.add(new Position(xi, y));
                }
            }
            goRight = !goRight;
            y += 1.0;
        }
        return spots;
    }

    @Override public int capacity()              { return spotPositions.size(); }
    @Override public int size()                  { return clients.size(); }
    @Override public int getOwnerId()            { return ownerId; }
    @Override public List<Client> getClients()   { return new ArrayList<>(clients); }

    @Override
    public Position getSpotPosition(int index) {
        return spotPositions.get(index);
    }

    @Override
    public Client getClientAt(int index) {
        if (index < 0 || index >= clients.size()) return null;
        return clients.get(index);
    }

    @Override
    public int enqueue(Client client) {
        int spotIndex = clients.size();
        clients.add(client);
        client.setQueueSpotIndex(spotIndex);
        return spotIndex;
    }

    /**
     * Removes the front client (index 0) and computes the advancement list for
     * all remaining clients (each shifts from index i+1 to index i).
     */
    @Override
    public List<QueueAdvancement> dequeue() {
        List<QueueAdvancement> advancements = new ArrayList<>();
        if (clients.isEmpty()) return advancements;

        clients.remove(0);

        for (int newIdx = 0; newIdx < clients.size(); newIdx++) {
            Client c      = clients.get(newIdx);
            int    oldIdx = newIdx + 1; // was at this index before removal
            advancements.add(new QueueAdvancement(c, oldIdx, newIdx));
            c.setQueueSpotIndex(newIdx);
        }
        return advancements;
    }
}
