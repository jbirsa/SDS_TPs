import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Generator {

    public static List<Particle> generate(
            int N,
            double L,
            double rMin,
            double rMax,
            boolean periodicBoundary,
            long seed
    ) {

        Random random = (seed >= 0) ? new Random(seed) : new Random();
        List<Particle> particles = new ArrayList<>(N);

        int maxAttempts = N * 1000;

        for (int i = 0; i < N; i++) {

            boolean placed = false;
            int attempts = 0;

            while (!placed) {

                if (attempts++ > maxAttempts) {
                    throw new IllegalStateException(
                            "Could not place particle " + i +
                                    " after " + maxAttempts +
                                    " attempts. Try reducing N or increasing L."
                    );
                }

                double radius = rMin + random.nextDouble() * (rMax - rMin);

                double x, y;

                if (periodicBoundary) {
                    // En PBC el centro puede estar en cualquier lugar
                    x = random.nextDouble() * L;
                    y = random.nextDouble() * L;
                } else {
                    // Sin PBC la partícula debe quedar completamente dentro
                    x = radius + random.nextDouble() * (L - 2 * radius);
                    y = radius + random.nextDouble() * (L - 2 * radius);
                }

                Particle candidate = new Particle(i, x, y, radius);

                if (!overlapsAny(candidate, particles, L, periodicBoundary)) {
                    particles.add(candidate);
                    placed = true;
                }
            }
        }

        return particles;
    }

    public static List<Particle> generate(int N, double L, double rMin, double rMax, boolean periodicBoundary) {
        return generate(N, L, rMin, rMax, periodicBoundary, -1);
    }

    private static boolean overlapsAny(
            Particle candidate,
            List<Particle> placed,
            double L,
            boolean periodicBoundary
    ) {

        for (Particle p : placed) {

            double dist = periodicBoundary
                    ? candidate.distanceToPBC(p, L)
                    : candidate.distanceTo(p);

            if (dist <= 0) {
                return true;
            }
        }

        return false;
    }
}