package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Generator {

    /**
     * Generates N non-overlapping particles randomly placed within [0, L] x [0, L].
     *
     * @param N      number of particles
     * @param L      side length of the square area
     * @param rMin   minimum radius (e.g. 0.23)
     * @param rMax   maximum radius (e.g. 0.26)
     * @param seed   random seed (-1 for no fixed seed)
     * @return list of N non-overlapping particles
     * @throws IllegalStateException if unable to place all particles after max attempts
     */
    public static List<Particle> generate(int N, double L, double rMin, double rMax, long seed) {
        Random random = (seed >= 0) ? new Random(seed) : new Random();
        List<Particle> particles = new ArrayList<>(N);

        int maxAttempts = N * 1000;
        int attempts = 0;

        for (int i = 0; i < N; i++) {
            boolean placed = false;

            while (!placed) {
                if (attempts++ > maxAttempts) {
                    throw new IllegalStateException(
                            "Could not place particle " + i + " after " + maxAttempts +
                                    " attempts. Try reducing N or increasing L."
                    );
                }

                double radius = rMin + random.nextDouble() * (rMax - rMin);
                // Keep particle fully inside the area
                double x = radius + random.nextDouble() * (L - 2 * radius);
                double y = radius + random.nextDouble() * (L - 2 * radius);

                Particle candidate = new Particle(i, x, y, radius);

                if (!overlapsAny(candidate, particles)) {
                    particles.add(candidate);
                    placed = true;
                }
            }
        }

        return particles;
    }

    /**
     * Convenience overload with no fixed seed.
     */
    public static List<Particle> generate(int N, double L, double rMin, double rMax) {
        return generate(N, L, rMin, rMax, -1);
    }

    /**
     * Checks if a candidate particle overlaps with any already-placed particle.
     * Overlap means edge-to-edge distance <= 0.
     */
    private static boolean overlapsAny(Particle candidate, List<Particle> placed) {
        for (Particle p : placed) {
            if (candidate.distanceTo(p) <= 0) {
                return true;
            }
        }
        return false;
    }
}