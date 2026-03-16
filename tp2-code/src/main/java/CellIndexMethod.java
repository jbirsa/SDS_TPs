import java.util.*;

public class CellIndexMethod {

    private final double L;
    private final int M;
    private final double rc;

    public CellIndexMethod(double L, int M, double rc) {
        this.L = L;
        this.M = M;
        this.rc = rc;
    }

    public Map<Integer, Set<Integer>> findNeighbors(List<Particle> particles) {
        Map<Integer, Set<Integer>> neighbors = new HashMap<>();
        for (Particle p : particles) {
            neighbors.put(p.getId(), new HashSet<>());
        }

        List<List<Integer>> grid = buildGrid(particles);

        for (int row = 0; row < M; row++) {
            for (int col = 0; col < M; col++) {
                List<Integer> cellParticles = grid.get(cellIndex(row, col));
                checkPairsInList(cellParticles, cellParticles, particles, neighbors, true);

                int[][] neighborOffsets = {
                        {0,  1},   // right
                        {1, -1},   // bottom-left
                        {1,  0},   // bottom
                        {1,  1},   // bottom-right
                };

                for (int[] offset : neighborOffsets) {
                    int nRow = row + offset[0];
                    int nCol = col + offset[1];
                    nRow = Math.floorMod(nRow, M);
                    nCol = Math.floorMod(nCol, M);
                    List<Integer> neighborParticles = grid.get(cellIndex(nRow, nCol));
                    checkPairsInList(cellParticles, neighborParticles, particles, neighbors, false);
                }
            }
        }

        return neighbors;
    }

    private List<List<Integer>> buildGrid(List<Particle> particles) {
        List<List<Integer>> grid = new ArrayList<>(M * M);
        for (int i = 0; i < M * M; i++) {
            grid.add(new ArrayList<>());
        }

        double cellSize = L / M;

        for (Particle p : particles) {
            int col = (int) (p.getX() / cellSize);
            int row = (int) (p.getY() / cellSize);
            col = Math.min(col, M - 1);
            row = Math.min(row, M - 1);
            grid.get(cellIndex(row, col)).add(p.getId());
        }

        return grid;
    }

    private void checkPairsInList(List<Integer> listA, List<Integer> listB, List<Particle> particles, Map<Integer, Set<Integer>> neighbors, boolean sameCell) {
        for (int idA : listA) {
            for (int idB : listB) {
                if (sameCell && idA >= idB) {
                    continue;
                }

                Particle pA = particles.get(idA);
                Particle pB = particles.get(idB);

                double dist = pA.distanceTo(pB, L);

                if (dist <= rc) {
                    neighbors.get(idA).add(idB);
                    neighbors.get(idB).add(idA);
                }
            }
        }
    }

    private int cellIndex(int row, int col) {
        return row * M + col;
    }

    public static int optimalM(double L, double rc, double rMax) {
        return Math.max(1, (int) Math.floor(L / (rc + 2 * rMax)));
    }
}
