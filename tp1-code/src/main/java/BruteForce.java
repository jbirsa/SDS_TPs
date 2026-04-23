import java.util.*;

public class BruteForce {

    private final double L;
    private final double rc;
    private final boolean periodicBoundary;

    /**
     * @param L                side length of the square area
     * @param rc               interaction radius (edge-to-edge cutoff)
     * @param periodicBoundary whether to apply periodic boundary conditions
     */
    public BruteForce(double L, double rc, boolean periodicBoundary) {
        this.L = L;
        this.rc = rc;
        this.periodicBoundary = periodicBoundary;
    }

    /**
     * Finds neighbors for all particles by checking every possible pair.
     * O(N^2) complexity — used as reference for correctness and benchmark comparison.
     *
     * @param particles list of particles
     * @return map from each particle's id to the set of neighbor particle ids
     */
    public Map<Integer, Set<Integer>> findNeighbors(List<Particle> particles) {
        Map<Integer, Set<Integer>> neighbors = new HashMap<>();
        for (Particle p : particles) {
            neighbors.put(p.getId(), new HashSet<>());
        }

        int n = particles.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Particle pA = particles.get(i);
                Particle pB = particles.get(j);

                double dist = periodicBoundary
                        ? pA.distanceToPBC(pB, L)
                        : pA.distanceTo(pB);

                if (dist <= rc) {
                    neighbors.get(pA.getId()).add(pB.getId());
                    neighbors.get(pB.getId()).add(pA.getId());
                }
            }
        }

        return neighbors;
    }

    /**
     * Verifies that two neighbor maps are identical.
     * Useful for checking CIM output against brute force.
     *
     * @return true if both maps contain the same neighbor sets for all particles
     */
    public static boolean verify(
            Map<Integer, Set<Integer>> expected,
            Map<Integer, Set<Integer>> actual) {

        if (!expected.keySet().equals(actual.keySet())) return false;

        for (int id : expected.keySet()) {
            if (!expected.get(id).equals(actual.get(id))) {
                System.err.printf(
                        "Mismatch for particle %d: expected %s, got %s%n",
                        id, expected.get(id), actual.get(id)
                );
                return false;
            }
        }
        return true;
    }
}