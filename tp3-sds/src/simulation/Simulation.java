package simulation;

import java.io.IOException;
import java.util.*;

/**
 * Discrete-event simulation of the Smart Queue system (Sistema 2 of TP3).
 *
 * <h2>Room layout</h2>
 * Square room of 30 × 30 m². Servers sit on the south wall (y = 0), evenly
 * spaced. Clients spawn at uniformly random positions in the interior.
 *
 * <h2>Event types and their effects</h2>
 * <ul>
 *   <li>{@link ClientArrivalEvent}: client spawns, is assigned to a queue,
 *       and walks to the last free spot. The next arrival is also scheduled.
 *   <li>{@link ClientArrivesAtQueueSpotEvent}: client physically reaches their
 *       assigned spot; if they are at the front and the server is available,
 *       they are immediately dispatched.
 *   <li>{@link ClientAdvancesInQueueEvent}: client finishes moving one spot
 *       forward after a service completion triggered the chain advance.
 *   <li>{@link ClientArrivesAtServerEvent}: client reaches the server; service
 *       begins and a {@link ServiceCompletionEvent} is scheduled.
 *   <li>{@link ServiceCompletionEvent}: service ends; client leaves the system;
 *       if the queue has a client physically at the front, they are dispatched
 *       and all others advance simultaneously.
 * </ul>
 *
 * <h2>Queue advance chain</h2>
 * A service completion (or a client dispatched from spot 0) triggers:
 * <ol>
 *   <li>The front client walks to the server (travel time = dist / 1 m s⁻¹).
 *   <li>All remaining clients shift one spot forward in parallel; each
 *       generates a {@link ClientAdvancesInQueueEvent} with its own travel time.
 * </ol>
 * Dispatch only fires when the front client's state is
 * {@link ClientState#IN_QUEUE} (physically at spot 0). If the client is still
 * in transit the server stays available; dispatch fires when the client arrives.
 */
public class Simulation {

    private final SimulationConfig config;
    private final Random           rng;

    private final List<Server>         servers    = new ArrayList<>();
    /**
     * Modality A: one queue per server (queues[i] belongs to servers[i]).
     * Modality B: queues[0] is the single shared queue.
     */
    private final List<Queue>          queues     = new ArrayList<>();
    private final Map<Integer, Client> clients    = new LinkedHashMap<>();
    private final EventQueue           eventQueue = new EventQueue();
    private final Map<Queue, LinkedHashSet<Integer>> approachingClients = new IdentityHashMap<>();

    private final Statistics   stats;
    private final OutputWriter output;
    private final ServerAssigner assigner;

    private double currentTime  = 0.0;
    private int    nextClientId = 1;

    // -------------------------------------------------------------------------
    public Simulation(SimulationConfig config) throws IOException {
        this.config   = config;
        this.rng      = new Random(config.seed);
        this.assigner = new ServerAssigner(rng, config.meanServiceTime);
        this.output   = new OutputWriter(config.outputFile);

        buildServersAndQueues();

        int numQueuesForStats = (config.modality == SimulationConfig.Modality.B)
                                ? 1 : config.numServers;
        this.stats = new Statistics(numQueuesForStats);
    }

    // -------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------

    private void buildServersAndQueues() {
        double stripWidth = config.roomSize / config.numServers;

        for (int i = 1; i <= config.numServers; i++) {
            double   sx        = (i - 0.5) * stripWidth; // centre of strip i
            Position serverPos = new Position(sx, 0.0);
            servers.add(new Server(i, serverPos));

            if (config.modality == SimulationConfig.Modality.A) {
                registerQueue(makeQueue(i, serverPos, stripWidth));
            }
        }

        if (config.modality == SimulationConfig.Modality.B) {
            // Shared queue anchored at the horizontal centre, lifted so the front
            // row sits a few metres above the servers for visual clarity.
            Position anchor = new Position(config.roomSize / 2.0, 4.0);
            registerQueue(makeQueue(-1, anchor, config.roomSize));
        }
    }

    private void registerQueue(Queue queue) {
        queues.add(queue);
        approachingClients.put(queue, new LinkedHashSet<>());
    }

    private Queue makeQueue(int ownerId, Position anchor, double stripWidth) {
        if (config.queueType == SimulationConfig.QueueType.SERPENTINE) {
            return new GuidedSerpentineQueue(ownerId, anchor, stripWidth);
        }
        return new FreeSpaceLineQueue(ownerId, anchor, stripWidth, config.roomSize, rng);
    }

    // -------------------------------------------------------------------------
    // Main loop
    // -------------------------------------------------------------------------

    public void run() {
        // Per-event frame writing is large for unstable, long runs. Disable via
        // -Dframes.enabled=false when only statistics are needed (studies 2.1–2.3);
        // keep enabled (default) for animation source runs.
        boolean writeFrames = !"false".equalsIgnoreCase(
                System.getProperty("frames.enabled", "true"));

        eventQueue.add(new ClientArrivalEvent(sampleExp(config.meanArrivalTime)));

        while (!eventQueue.isEmpty()) {
            Event event = eventQueue.poll();
            if (event.time > config.simulationTime) break;

            currentTime = event.time;
            updateMovingClients(currentTime);
            dispatchEvent(event);
            recordQueueLengthSample();
            if (writeFrames) {
                output.writeFrame(currentTime, clients.values(), servers);
            }
        }

        output.writeStatistics(config, stats, config.simulationTime);
        output.close();
    }

    // -------------------------------------------------------------------------
    // Event dispatch
    // -------------------------------------------------------------------------

    private void dispatchEvent(Event e) {
        if      (e instanceof ClientArrivalEvent)              processClientArrival();
        else if (e instanceof ClientArrivesAtQueueSpotEvent)   processClientArrivesAtQueueSpot((ClientArrivesAtQueueSpotEvent) e);
        else if (e instanceof ClientAdvancesInQueueEvent)      processClientAdvancesInQueue((ClientAdvancesInQueueEvent) e);
        else if (e instanceof ClientArrivesAtServerEvent)      processClientArrivesAtServer((ClientArrivesAtServerEvent) e);
        else if (e instanceof ServiceCompletionEvent)          processServiceCompletion((ServiceCompletionEvent) e);
    }

    // -------------------------------------------------------------------------
    // Event handlers
    // -------------------------------------------------------------------------

    /** A new client spawns at a random interior position and joins a queue. */
    private void processClientArrival() {
        double margin = 1.0;
        double x = margin + rng.nextDouble() * (config.roomSize - 2 * margin);
        double y = margin + rng.nextDouble() * (config.roomSize - 2 * margin);
        Position spawn = new Position(x, y);

        Client client = new Client(nextClientId++, spawn, currentTime);
        clients.put(client.id, client);

        Queue queue     = chooseQueue(client);
        client.setAssignedServerId(queue.getOwnerId()); // -1 if shared
        approachingClients.get(queue).add(client.id);
        retargetApproachingClient(queue, client);

        // Schedule next arrival
        eventQueue.add(new ClientArrivalEvent(currentTime + sampleExp(config.meanArrivalTime)));
    }

    /**
     * Client reaches their assigned queue spot. Updates their physical position
     * and — if they are now at the front with an available server — dispatches them.
     */
    private void processClientArrivesAtQueueSpot(ClientArrivesAtQueueSpotEvent event) {
        Client client = clients.get(event.clientId);
        if (client == null) return;
        if (event.movementVersion != client.getMovementVersion()) return;

        Queue queue = queueForClient(client);
        if (queue == null) return;
        if (!approachingClients.get(queue).remove(client.id)) return;

        int idx = queue.enqueue(client);
        client.finishMovement();
        client.setPosition(queue.getSpotPosition(idx));
        client.setState(ClientState.IN_QUEUE);
        retargetApproachingClients(queue);

        if (idx == 0) {
            tryDispatch(queue);
        }
    }

    /**
     * Client arrives at a new (closer) queue spot after the advance chain fired.
     * <p>A client may have <em>multiple</em> advance events pending simultaneously
     * (one per service-completion that shifted the queue while they were in transit).
     * Guards below ensure stale events are silently discarded.
     */
    private void processClientAdvancesInQueue(ClientAdvancesInQueueEvent event) {
        Client client = clients.get(event.clientId);
        if (client == null) return;
        if (event.movementVersion != client.getMovementVersion()) return;
        if (isAlreadyDispatched(client)) return;

        Queue queue = queueForClient(client);
        if (queue == null) return;

        int idx = client.getQueueSpotIndex();
        if (idx < 0) return; // safety: index invalidated by earlier dispatch
        if (event.newSpotIndex != idx) return;
        client.finishMovement();
        client.setPosition(queue.getSpotPosition(idx));
        client.setState(ClientState.IN_QUEUE);

        if (idx == 0) {
            tryDispatch(queue);
        }
    }

    /** Returns true when a client is walking to or already at a server (not in queue). */
    private boolean isAlreadyDispatched(Client client) {
        ClientState s = client.getState();
        return s == ClientState.WALKING_TO_SERVER || s == ClientState.BEING_SERVED;
    }

    /** Client walks into the server; service begins; completion event scheduled. */
    private void processClientArrivesAtServer(ClientArrivesAtServerEvent event) {
        Client client = clients.get(event.clientId);
        Server server = servers.get(event.serverId - 1);
        if (client == null) return;
        if (event.movementVersion != client.getMovementVersion()) return;
        if (client.getState() != ClientState.WALKING_TO_SERVER) return;
        client.finishMovement();

        // Update server assignment for modality B (was -1 while in shared queue)
        client.setAssignedServerId(server.id);
        client.setQueueSpotIndex(-1);
        server.startServing(client);

        double serviceTime = sampleExp(config.meanServiceTime);
        eventQueue.add(new ServiceCompletionEvent(currentTime + serviceTime, server.id, client.id));
    }

    /** Service ends; client departs; next client dispatched from queue if ready. */
    private void processServiceCompletion(ServiceCompletionEvent event) {
        Server server = servers.get(event.serverId - 1);
        Client served = server.finishServing();
        if (served == null || served.id != event.clientId) return; // stale event

        stats.recordDeparture(served.arrivalTime, currentTime);
        clients.remove(served.id);

        Queue queue = queueForServer(server);
        // Pass the freed server as hint: for Modality B this prevents handing the
        // next client to server 1 when the actual freed server is still available.
        tryDispatch(queue, server);
    }

    // -------------------------------------------------------------------------
    // Core dispatch logic
    // -------------------------------------------------------------------------

    /**
     * If the front client of {@code queue} is physically present (IN_QUEUE) at
     * spot 0 and an appropriate server is available, dispatches them:
     * <ol>
     *   <li>Removes them from the queue.
     *   <li>Reserves the server and schedules their walk.
     *   <li>Schedules the parallel advance chain for all remaining clients.
     * </ol>
     * If the front client is still in transit, this method does nothing — dispatch
     * will be retried when they arrive.
     */
    /** Convenience overload: no preferred server. */
    private void tryDispatch(Queue queue) {
        tryDispatch(queue, null);
    }

    /**
     * @param preferredServer If non-null and available, use it preferentially
     *                        (avoids handing work to server 1 when a different
     *                        server just freed up in Modality B).
     */
    private void tryDispatch(Queue queue, Server preferredServer) {
        if (queue == null || queue.size() == 0) return;

        Client front = queue.getClientAt(0);
        if (front == null || front.getState() != ClientState.IN_QUEUE) return;

        Server server;
        if (preferredServer != null && preferredServer.isAvailable()) {
            server = preferredServer;
        } else {
            server = availableServerFor(queue);
        }
        if (server == null) return;

        // Reserve server immediately so no other event can double-dispatch
        server.reserveFor(front.id);
        front.setState(ClientState.WALKING_TO_SERVER);

        // Remove front from queue; get advance list for remaining clients
        List<QueueAdvancement> advances = queue.dequeue();
        retargetApproachingClients(queue);

        // Walk front client from their current position (spot 0) to the server
        double travel = front.getPosition().distanceTo(server.position) / config.walkingSpeed;
        int movementVersion = front.startMovement(server.position, currentTime, currentTime + travel);
        eventQueue.add(new ClientArrivesAtServerEvent(currentTime + travel, front.id, server.id, movementVersion));

        // Parallel advance chain: every remaining client moves one spot forward simultaneously
        for (QueueAdvancement adv : advances) {
            Position to = queue.getSpotPosition(adv.newSpotIndex);
            double advTime = adv.client.getPosition().distanceTo(to) / config.walkingSpeed;
            adv.client.setState(ClientState.ADVANCING_IN_QUEUE);
            int advVersion = adv.client.startMovement(to, currentTime, currentTime + advTime);
            eventQueue.add(new ClientAdvancesInQueueEvent(
                currentTime + advTime, adv.client.id, adv.newSpotIndex, advVersion));
        }
    }

    // -------------------------------------------------------------------------
    // Routing helpers
    // -------------------------------------------------------------------------

    /** Weighted-random queue selection (Modality A) or shared queue (Modality B). */
    private Queue chooseQueue(Client client) {
        if (config.modality == SimulationConfig.Modality.B) {
            return queues.get(0);
        }
        int[] approachingCounts = new int[queues.size()];
        for (int i = 0; i < queues.size(); i++) {
            approachingCounts[i] = approachingClients.get(queues.get(i)).size();
        }
        Server chosen = assigner.assign(client, servers, queues, approachingCounts);
        return queues.get(chosen.id - 1);
    }

    private Queue queueForClient(Client client) {
        if (config.modality == SimulationConfig.Modality.B) return queues.get(0);
        int sid = client.getAssignedServerId();
        if (sid < 1 || sid > queues.size()) return null;
        return queues.get(sid - 1);
    }

    private Queue queueForServer(Server server) {
        if (config.modality == SimulationConfig.Modality.B) return queues.get(0);
        return queues.get(server.id - 1);
    }

    /**
     * Returns an available (free and not reserved) server for the given queue.
     * Modality A: the specific owning server. Modality B: any available server.
     */
    private Server availableServerFor(Queue queue) {
        if (config.modality == SimulationConfig.Modality.A) {
            Server s = servers.get(queue.getOwnerId() - 1);
            return s.isAvailable() ? s : null;
        }
        for (Server s : servers) {
            if (s.isAvailable()) return s;
        }
        return null;
    }

    private void retargetApproachingClients(Queue queue) {
        List<Integer> clientIds = new ArrayList<>(approachingClients.get(queue));
        for (int clientId : clientIds) {
            Client client = clients.get(clientId);
            if (client == null) {
                approachingClients.get(queue).remove(clientId);
                continue;
            }
            retargetApproachingClient(queue, client);
        }
    }

    private void retargetApproachingClient(Queue queue, Client client) {
        int targetIndex = queue.size();
        if (targetIndex >= queue.capacity()) {
            // Queue is at full capacity; keep the client walking toward their last
            // known target rather than trying to access a non-existent spot.
            return;
        }
        Position target = queue.getSpotPosition(targetIndex);
        double travel = client.getPosition().distanceTo(target) / config.walkingSpeed;
        client.setState(ClientState.WALKING_TO_QUEUE_SPOT);
        int movementVersion = client.startMovement(target, currentTime, currentTime + travel);
        eventQueue.add(new ClientArrivesAtQueueSpotEvent(currentTime + travel, client.id, movementVersion));
    }

    // -------------------------------------------------------------------------
    // Statistics & sampling
    // -------------------------------------------------------------------------

    private void recordQueueLengthSample() {
        int[] lengths = new int[queues.size()];
        for (int i = 0; i < queues.size(); i++) {
            lengths[i] = queues.get(i).size();
        }
        stats.recordQueueLengths(currentTime, lengths);
    }

    private void updateMovingClients(double time) {
        for (Client client : clients.values()) {
            client.updatePositionAt(time);
        }
    }

    private double sampleExp(double mean) {
        return -mean * Math.log(1.0 - rng.nextDouble());
    }
}
