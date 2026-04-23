package simulation;

import java.io.IOException;
import java.util.Locale;

/**
 * Entry point. Runs a single simulation scenario.
 *
 * <p>Usage:
 * <pre>
 *   javac -d out src/simulation/*.java
 *   java  -cp out simulation.Main [t1] [t2] [k] [modality] [queueType] [simTime] [seed]
 * </pre>
 *
 * <p>All arguments are optional; unspecified ones fall back to the defaults
 * hardcoded below. Arguments must be provided in order (no named flags).
 *
 * <p>Examples:
 * <pre>
 *   java -cp out simulation.Main                          # all defaults
 *   java -cp out simulation.Main 1.0 3.0 3 A FREE 300 42
 *   java -cp out simulation.Main 0.5 2.0 5 B SERPENTINE
 * </pre>
 */
public class Main {

    // ── Default parameters ────────────────────────────────────────────────────
    private static final double   DEFAULT_T1        = 1.0;
    private static final double   DEFAULT_T2        = 3.0;
    private static final int      DEFAULT_K         = 1;
    private static final String   DEFAULT_MODALITY  = "A";
    private static final String   DEFAULT_QUEUE_TYPE = "FREE";
    private static final double   DEFAULT_SIM_TIME  = 50.0;
    private static final long     DEFAULT_SEED      = 42L;
    // ─────────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {
        double t1       = args.length > 0 ? Double.parseDouble(args[0])  : DEFAULT_T1;
        double t2       = args.length > 1 ? Double.parseDouble(args[1])  : DEFAULT_T2;
        int    k        = args.length > 2 ? Integer.parseInt(args[2])    : DEFAULT_K;
        String modStr   = args.length > 3 ? args[3].toUpperCase()        : DEFAULT_MODALITY;
        String qStr     = args.length > 4 ? args[4].toUpperCase()        : DEFAULT_QUEUE_TYPE;
        double simTime  = args.length > 5 ? Double.parseDouble(args[5])  : DEFAULT_SIM_TIME;
        long   seed     = args.length > 6 ? Long.parseLong(args[6])      : DEFAULT_SEED;

        SimulationConfig.Modality  modality  = SimulationConfig.Modality.valueOf(modStr);
        SimulationConfig.QueueType queueType = SimulationConfig.QueueType.valueOf(qStr);

        String filename = String.format(Locale.US, "out_%s_%s_t1=%.2f_t2=%.2f_k=%d.txt",
                modality.name(), queueType.name(), t1, t2, k);

        SimulationConfig cfg = new SimulationConfig.Builder()
                .meanArrivalTime(t1)
                .meanServiceTime(t2)
                .numServers(k)
                .modality(modality)
                .queueType(queueType)
                .simulationTime(simTime)
                .outputFile(filename)
                .seed(seed)
                .build();

        System.out.printf(Locale.US, "Running: t1=%.2f  t2=%.2f  k=%d  %s  %s  seed=%d  simTime=%.1f  → %s%n",
                t1, t2, k, modality, queueType, seed, simTime, filename);

        new Simulation(cfg).run();

        System.out.println("Done. Output written to tp3-output/" + filename);
    }
}
