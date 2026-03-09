import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Main {

    // Default parameters
    private static final int    N_DEFAULT     = 500;
    private static final double L_DEFAULT     = 20.0;
    private static final int    M_DEFAULT     = -1;   // -1 = auto-compute optimal
    private static final double RC_DEFAULT    = 1.0;
    private static final double R_MIN         = 0.23;
    private static final double R_MAX         = 0.26;
    private static final boolean PBC_DEFAULT  = false;
    private static final int    HIGHLIGHT_ID  = 5;
    private static final int    RUNS_DEFAULT  = 10;

    private static final String OUTPUT_PATH = "/Users/josefinagonzalezcornet/Desktop/1c2026/sds/SDS_TPs/tp1-output/";

    public static void main(String[] args) throws IOException {

        // ----------------------------------------------------------------
        // 1. Parse arguments (all optional, fall back to defaults)
        //    Usage: java Main [N] [L] [M] [rc] [pbc] [highlightId] [seed] [runs]
        //    Example: java Main 200 20 5 1.0 false 0 42 5
        // ----------------------------------------------------------------
        int     N           = argInt(args, 0, N_DEFAULT);
        double  L           = argDouble(args, 1, L_DEFAULT);
        int     M           = argInt(args, 2, M_DEFAULT);
        double  rc          = argDouble(args, 3, RC_DEFAULT);
        boolean pbc         = argBool(args, 4, PBC_DEFAULT);
        int     highlightId = argInt(args, 5, HIGHLIGHT_ID);
        long    seed        = argLong(args, 6, -1L);
        int runs = Math.max(1, argInt(args, 7, RUNS_DEFAULT));

        // Auto-compute optimal M if not provided
        if (M <= 0) {
            M = CellIndexMethod.optimalM(L, rc, R_MAX);
            System.out.printf("Auto-computed optimal M = %d%n", M);
        }

        System.out.printf("Parameters: N=%d  L=%.1f  M=%d  rc=%.2f  PBC=%s  highlight=%d%n",
                N, L, M, rc, pbc, highlightId);
        System.out.println("--------------------------------------------------");

        System.out.printf("Simulations per configuration: %d%n", runs);

        CellIndexMethod cim = new CellIndexMethod(L, M, rc, pbc);
        BruteForce bf = new BruteForce(L, rc, pbc);

        double totalCimMs = 0.0;
        double totalBfMs = 0.0;
        boolean allMatches = true;

        List<Particle> lastParticles = null;
        Map<Integer, Set<Integer>> lastNeighbors = null;

        for (int run = 0; run < runs; run++) {
            long runSeed = (seed >= 0) ? seed + run : -1;
            List<Particle> particles;
            try {
                particles = Generator.generate(N, L, R_MIN, R_MAX, runSeed > 0);
            } catch (IllegalStateException e) {
                System.err.println("ERROR generating particles: " + e.getMessage());
                return;
            }
            long cimStart = System.nanoTime();
            Map<Integer, Set<Integer>> cimNeighbors = cim.findNeighbors(particles);
            long cimEnd   = System.nanoTime();
            double cimMs  = (cimEnd - cimStart) / 1_000_000.0;

            long bfStart = System.nanoTime();
            Map<Integer, Set<Integer>> bfNeighbors = bf.findNeighbors(particles);
            long bfEnd   = System.nanoTime();
            double bfMs  = (bfEnd - bfStart) / 1_000_000.0;

            boolean match = BruteForce.verify(bfNeighbors, cimNeighbors);
            allMatches &= match;
            if (!match) {
                System.err.printf("Run %d: CIM vs BruteForce MISMATCH%n", run + 1);
            }

            totalCimMs += cimMs;
            totalBfMs += bfMs;

            if (run == runs - 1) {
                lastParticles = particles;
                lastNeighbors = cimNeighbors;
            }
        }

        double avgCim = totalCimMs / runs;
        double avgBf  = totalBfMs / runs;

        System.out.println("CIM vs BruteForce match: " + (allMatches ? "OK" : "MISMATCH"));
        System.out.printf("Average CIM time:        %.4f ms%n", avgCim);
        System.out.printf("Average BruteForce time: %.4f ms%n", avgBf);
        System.out.println("--------------------------------------------------");

        if (lastParticles != null && lastNeighbors != null) {
            if (highlightId >= 0 && highlightId < N) {
                Set<Integer> highlighted = lastNeighbors.get(highlightId);
                System.out.printf("Neighbors of particle %d (last run): %s%n", highlightId, highlighted);
            }

            writeStaticFile(OUTPUT_PATH + "static.txt",  lastParticles, L);
            writeDynamicFile(OUTPUT_PATH + "dynamic.txt", lastParticles);
            writeNeighborsFile(OUTPUT_PATH + "neighbors.txt", lastNeighbors);
            writeOvitoFile(OUTPUT_PATH + "ovito.xyz", lastParticles, lastNeighbors, highlightId);

            System.out.println("Output files written (last run): static.txt, dynamic.txt, neighbors.txt, ovito.xyz");
        }
    }

    // ----------------------------------------------------------------
    // File writers
    // ----------------------------------------------------------------

    /** Static file: N, L, then per-particle radius and a placeholder property. */
    private static void writeStaticFile(String path, List<Particle> particles, double L) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println(particles.size());
            pw.println(L);
            for (Particle p : particles) {
                pw.printf(Locale.US, "%.4f %n", p.getRadius());
            }
        }
    }

    /** Dynamic file: single timestep t=0, then x y vx vy per particle (vx=vy=0). */
    private static void writeDynamicFile(String path, List<Particle> particles) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println(0); // t0
            for (Particle p : particles) {
                pw.printf(Locale.US, "%.4f %.4f 0.0 0.0%n", p.getX(), p.getY());
            }
        }
    }

    /** Neighbors file: one line per particle listing its neighbor ids. */
    private static void writeNeighborsFile(String path, Map<Integer, Set<Integer>> neighbors) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            List<Integer> ids = new ArrayList<>(neighbors.keySet());
            Collections.sort(ids);
            for (int id : ids) {
                List<Integer> sorted = new ArrayList<>(neighbors.get(id));
                Collections.sort(sorted);
                pw.printf("%d %s%n", id, sorted.toString().replaceAll("[\\[\\],]", "").trim());
            }
        }
    }

    /**
     * Ovito-compatible XYZ file.
     * Color encoding via "type" column:
     *   2 = highlighted particle
     *   1 = neighbor of highlighted
     *   0 = regular particle
     */
    private static void writeOvitoFile(
            String path,
            List<Particle> particles,
            Map<Integer, Set<Integer>> neighbors,
            int highlightId) throws IOException {

        Set<Integer> highlightNeighbors = neighbors.getOrDefault(highlightId, Collections.emptySet());

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println(particles.size());
            pw.println("Properties=id:I:1:pos:R:2:radius:R:1:type:I:1");
            for (Particle p : particles) {
                int type;
                if (p.getId() == highlightId)           type = 2;
                else if (highlightNeighbors.contains(p.getId())) type = 1;
                else                                              type = 0;

                pw.printf(Locale.US, "%d %.4f %.4f %.4f %d%n",
                        p.getId(), p.getX(), p.getY(), p.getRadius(), type);
            }
        }
    }

    // ----------------------------------------------------------------
    // Argument helpers
    // ----------------------------------------------------------------

    private static int argInt(String[] args, int i, int def) {
        return (args.length > i) ? Integer.parseInt(args[i]) : def;
    }

    private static double argDouble(String[] args, int i, double def) {
        return (args.length > i) ? Double.parseDouble(args[i]) : def;
    }

    private static boolean argBool(String[] args, int i, boolean def) {
        return (args.length > i) ? Boolean.parseBoolean(args[i]) : def;
    }

    private static long argLong(String[] args, int i, long def) {
        return (args.length > i) ? Long.parseLong(args[i]) : def;
    }
}
