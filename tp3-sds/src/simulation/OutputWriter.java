package simulation;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

/**
 * Writes one text frame per simulation event and a summary block at the end.
 *
 * <p>Frame format (one frame per event):
 * <pre>
 * TIME 1.234567
 * CLIENT 1 12.5000 8.3000 WALKING_TO_QUEUE_SPOT -1 -1 255 100 50
 * CLIENT 2  5.0000 2.0000 IN_QUEUE               1  0 100 255 50
 * SERVER 1  5.0000 0.0000 BUSY 2
 * SERVER 2 15.0000 0.0000 FREE -1
 * ---
 * </pre>
 * CLIENT columns: id  x  y  state  assignedServerId  queueSpotIndex  r  g  b
 * SERVER columns: id  x  y  (BUSY|FREE)  currentClientId
 *
 * <p>Statistics are appended after all frames under a {@code STATS} header.
 * The {@code QUEUE_TIMESERIES} section records the total queue length sampled
 * at regular intervals so that post-processing can distinguish stable runs
 * (flat series) from unstable ones (growing series) and compute the growth rate.
 */
public class OutputWriter implements Closeable {

    private final PrintWriter writer;

    public OutputWriter(String filePath) throws IOException {
        Path outputPath = resolveOutputPath(filePath);
        Files.createDirectories(outputPath.getParent());
        this.writer = new PrintWriter(new BufferedWriter(new FileWriter(outputPath.toFile())));
    }

    private static Path resolveOutputPath(String filePath) {
        Path requested = Paths.get(filePath);
        if (requested.isAbsolute()) {
            return requested.normalize();
        }
        if (requested.getParent() != null) {
            return requested.toAbsolutePath().normalize();
        }
        return resolveOutputDir().resolve(requested.getFileName());
    }

    private static Path resolveOutputDir() {
        String override = System.getProperty("output.dir");
        if (override != null && !override.isBlank()) {
            return Paths.get(override).toAbsolutePath().normalize();
        }

        Path current = Paths.get("").toAbsolutePath().normalize();
        while (current != null) {
            Path candidate = current.resolve("tp3-output");
            if (Files.isDirectory(candidate) || Files.isDirectory(current.resolve("tp3-sds"))) {
                return candidate;
            }
            current = current.getParent();
        }

        return Paths.get("").toAbsolutePath().normalize().resolve("tp3-output");
    }

    /** Writes one snapshot frame representing the system state after an event. */
    public void writeFrame(double time,
                           Collection<Client> clients,
                           List<Server> servers) {
        writer.printf(Locale.US, "TIME %.6f%n", time);

        for (Client c : clients) {
            writer.printf(Locale.US, "CLIENT %d %.4f %.4f %s %d %d %d %d %d%n",
                c.id,
                c.getPosition().x,
                c.getPosition().y,
                c.getState().name(),
                c.getAssignedServerId(),
                c.getQueueSpotIndex(),
                c.rgb[0], c.rgb[1], c.rgb[2]);
        }

        for (Server s : servers) {
            int currentClientId = s.isBusy() ? s.getCurrentClient().id : -1;
            writer.printf(Locale.US, "SERVER %d %.4f %.4f %s %d%n",
                s.id,
                s.position.x,
                s.position.y,
                s.isBusy() ? "BUSY" : "FREE",
                currentClientId);
        }

        writer.println("---");
    }

    /**
     * Appends a statistics summary at the end of the output file.
     *
     * <p>Sections written:
     * <ul>
     *   <li>{@code STATS} — scalar metrics (t1, t2, k, average queue lengths, etc.)
     *   <li>{@code QUEUE_TIMESERIES} — total queue length sampled every 10 s of
     *       simulated time; used by post-processing to detect stable vs. unstable
     *       regimes and compute the queue growth rate when needed.
     *   <li>{@code PERMANENCE_TIMES} — per-client (arrival, departure, duration) triples.
     * </ul>
     */
    public void writeStatistics(SimulationConfig config,
                                Statistics stats,
                                double endTime) {
        writer.println("STATS");
        writer.printf(Locale.US, "t1=%.4f t2=%.4f k=%d modality=%s queueType=%s%n",
            config.meanArrivalTime, config.meanServiceTime,
            config.numServers, config.modality.name(), config.queueType.name());
        writer.printf(Locale.US, "simulationTime=%.2f clientsServed=%d%n",
            endTime, stats.getClientsServed());

        double[] avgQL = stats.computeAverageQueueLengths(endTime);
        for (int i = 0; i < avgQL.length; i++) {
            writer.printf(Locale.US, "avgQueueLength[%d]=%.4f%n", i, avgQL[i]);
        }

        writer.printf(Locale.US, "avgPermanenceTime=%.4f%n",
            stats.computeAveragePermanenceTime());

        // ── Queue-length time series ──────────────────────────────────────────
        List<double[]> timeSeries = stats.getQueueTimeSeries();
        if (!timeSeries.isEmpty()) {
            writer.println("QUEUE_TIMESERIES");
            for (double[] pt : timeSeries) {
                writer.printf(Locale.US, "%.1f %.4f%n", pt[0], pt[1]);
            }
        }

        // ── Per-client permanence times ───────────────────────────────────────
        writer.println("PERMANENCE_TIMES");
        for (double[] pt : stats.getPermanenceTimes()) {
            writer.printf(Locale.US, "%.6f %.6f %.6f%n", pt[0], pt[1], pt[1] - pt[0]);
        }
    }

    @Override
    public void close() {
        writer.flush();
        writer.close();
    }
}
