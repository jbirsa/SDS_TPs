package simulation;

import java.util.List;
import java.util.Random;

/**
 * Assigns arriving clients to servers using a probabilistic draw based on
 * an estimated total time cost.
 *
 * <h2>Assignment logic</h2>
 * For each server i, we estimate:
 *
 *   cost_i = walkingTime_i + waitingTime_i
 *
 * where:
 *
 *   walkingTime_i = distance(client, queue target for server i) / WALKING_SPEED
 *   waitingTime_i = peopleAhead_i * MEAN_SERVICE_TIME
 *
 * Then we convert those costs into probabilities with a negative softmax:
 *
 *   weight_i = exp(-TEMPERATURE * cost_i)
 *
 * Finally, we perform a weighted random draw.
 *
 * <h2>Why this is aligned with the statement</h2>
 * The statement requires the assignment to be probabilistic and to depend on:
 *   - the distance between the client and each server / queue
 *   - the amount of people in the queue associated with each server
 *
 * This heuristic uses exactly those two factors, but expressed in a common,
 * physically interpretable unit: time.
 */
public class ServerAssigner {

    /**
     * Walking speed fixed by the statement: 1 m/s.
     */
    private static final double WALKING_SPEED = 1.0;

    /**
     * Softmax sharpness.
     *
     * Higher values make the choice more deterministic.
     * Lower values make the choice more random.
     *
     * With TEMPERATURE = 1.0, servers with lower estimated total time
     * are favored, but all servers still retain some probability.
     */
    private static final double TEMPERATURE = 1.0;

    private final Random random;

    /**
     * Mean service time t2 (in seconds), required to estimate expected waiting time.
     */
    private final double meanServiceTime;

    /**
     * @param random random generator used for the weighted draw
     * @param meanServiceTime mean service time t2 (seconds)
     */
    public ServerAssigner(Random random, double meanServiceTime) {
        this.random = random;
        this.meanServiceTime = meanServiceTime;
    }

    /**
     * Selects one server for the arriving client via a weighted random draw.
     *
     * @param client the arriving client
     * @param servers ordered list of servers
     * @param queues one queue per server (same order as servers)
     * @param approachingCounts number of clients currently walking toward each queue
     * @return the chosen server
     */
    public Server assign(Client client,
                         List<Server> servers,
                         List<Queue> queues,
                         int[] approachingCounts) {
        int n = servers.size();

        double[] costs = new double[n];
        double[] weights = new double[n];
        double weightSum = 0.0;

        for (int i = 0; i < n; i++) {
            Server server = servers.get(i);
            Queue queue = queues.get(i);

            // Distance from client to the server.
            // If your implementation already knows the exact "last free spot"
            // of the queue, it would be even better to use that position instead.
            double distance = client.getPosition().distanceTo(server.position);

            // Time needed to reach that server/queue region.
            double walkingTime = distance / WALKING_SPEED;

            // Effective number of people ahead:
            // - clients already standing in the queue
            // - clients currently walking toward this queue
            // - the one being served, if the server is busy
            int peopleAhead = queue.size()
                    + approachingCounts[i]
                    + (server.isAvailable() ? 0 : 1);

            // Expected waiting time assuming each person ahead consumes, on average, t2 seconds.
            double waitingTime = peopleAhead * meanServiceTime;

            // Total estimated cost in seconds.
            double cost = walkingTime + waitingTime;
            costs[i] = cost;

            // Negative softmax: lower cost => higher weight.
            double weight = Math.exp(-TEMPERATURE * cost);
            weights[i] = weight;
            weightSum += weight;
        }

        // Weighted random draw.
        double r = random.nextDouble() * weightSum;
        double cumulative = 0.0;

        for (int i = 0; i < n; i++) {
            cumulative += weights[i];
            if (r < cumulative) {
                return servers.get(i);
            }
        }

        // Numerical safety fallback.
        return servers.get(n - 1);
    }
}