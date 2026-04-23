package simulation;

import java.util.ArrayList;
import java.util.List;

/**
 * Accumulates time-weighted queue-length samples, a periodic queue-length time
 * series, and per-client permanence times.
 *
 * <h2>Queue-length average</h2>
 * Sampled at every event via {@link #recordQueueLengths}. Time weighting (area
 * under the queue-length curve) ensures the average is not biased by
 * high-frequency events.
 *
 * <h2>Queue-length time series</h2>
 * To distinguish between stable (stationary) and unstable (unbounded growth)
 * regimes, the total queue length is recorded at fixed {@link #SAMPLE_INTERVAL}
 * second intervals throughout the simulation.  Post-processing can then fit a
 * linear trend to the second half of the series and report either the average
 * queue length (stable) or the growth rate in clients/s (unstable).
 *
 * <h2>Permanence times</h2>
 * Recorded when a client departs via {@link #recordDeparture}.
 */
public class Statistics {

    /** Simulation-time interval (seconds) between queue-length time-series samples. */
    private static final double SAMPLE_INTERVAL = 10.0;

    // ── Time-weighted queue length ────────────────────────────────────────────
    private final int     numQueues;
    private final double[] weightedSum;  // integral of queue length over time
    private final int[]    lastLength;   // queue length at the last event
    private double         lastEventTime;

    // ── Periodic queue-length time series ────────────────────────────────────
    private double         nextSampleTime = SAMPLE_INTERVAL;
    /**
     * Each entry is {simTime, totalQueueLength} recorded at multiples of
     * {@link #SAMPLE_INTERVAL}.
     */
    private final List<double[]> queueTimeSeries = new ArrayList<>();

    // ── Permanence times ──────────────────────────────────────────────────────
    private final List<double[]> permanenceTimes = new ArrayList<>(); // (arrival, departure)
    private int clientsServed = 0;

    public Statistics(int numQueues) {
        this.numQueues     = numQueues;
        this.weightedSum   = new double[numQueues];
        this.lastLength    = new int[numQueues];
        this.lastEventTime = 0.0;
    }

    /**
     * Must be called at every event.  Records the time-weighted contribution of
     * the <em>previous</em> queue lengths, then updates {@code lastLength} to the
     * new values.  Also takes periodic snapshots for the time series.
     *
     * @param currentTime current simulation time
     * @param lengths     queue lengths at this instant (one per queue)
     */
    public void recordQueueLengths(double currentTime, int[] lengths) {
        double dt = currentTime - lastEventTime;

        // Periodic time-series sampling: between [lastEventTime, currentTime)
        // the queue had length lastLength (step function between events).
        while (nextSampleTime <= currentTime) {
            double totalQL = 0.0;
            for (int l : lastLength) totalQL += l;
            queueTimeSeries.add(new double[]{nextSampleTime, totalQL});
            nextSampleTime += SAMPLE_INTERVAL;
        }

        // Time-weighted accumulation
        for (int i = 0; i < numQueues; i++) {
            weightedSum[i] += lastLength[i] * dt;
            lastLength[i]   = lengths[i];
        }
        lastEventTime = currentTime;
    }

    /**
     * Records that a client has left the system.
     *
     * @param arrivalTime   time the client spawned
     * @param departureTime time the service completed
     */
    public void recordDeparture(double arrivalTime, double departureTime) {
        permanenceTimes.add(new double[]{arrivalTime, departureTime});
        clientsServed++;
    }

    /**
     * Finalizes accumulation up to {@code endTime}, flushes any remaining
     * time-series samples, and returns average queue lengths.
     */
    public double[] computeAverageQueueLengths(double endTime) {
        // Flush remaining samples up to endTime
        while (nextSampleTime <= endTime) {
            double totalQL = 0.0;
            for (int l : lastLength) totalQL += l;
            queueTimeSeries.add(new double[]{nextSampleTime, totalQL});
            nextSampleTime += SAMPLE_INTERVAL;
        }

        // Flush the last partial interval
        double dt = endTime - lastEventTime;
        double[] averages = new double[numQueues];
        for (int i = 0; i < numQueues; i++) {
            double total = weightedSum[i] + lastLength[i] * dt;
            averages[i] = (endTime > 0) ? total / endTime : 0.0;
        }
        return averages;
    }

    /** Average permanence time across all served clients. */
    public double computeAveragePermanenceTime() {
        if (permanenceTimes.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double[] pt : permanenceTimes) sum += pt[1] - pt[0];
        return sum / permanenceTimes.size();
    }

    public List<double[]> getPermanenceTimes()  { return permanenceTimes; }
    public List<double[]> getQueueTimeSeries()  { return queueTimeSeries; }
    public int            getClientsServed()    { return clientsServed; }
}
