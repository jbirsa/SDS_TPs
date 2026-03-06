package org.example;

import java.util.*;

public class CellIndexMethod {

    private final double L;
    private final int M;
    private final double rc;
    private final boolean periodicBoundary;

    /**
     * @param L                side length of the square area
     * @param M                number of cells per side (grid is M x M)
     * @param rc               interaction radius (center-to-center cutoff before subtracting radii)
     * @param periodicBoundary whether to apply periodic boundary conditions
     */
    public CellIndexMethod(double L, int M, double rc, boolean periodicBoundary) {
        this.L = L;
        this.M = M;
        this.rc = rc;
        this.periodicBoundary = periodicBoundary;
    }

    /**
     * Computes neighbors for all particles.
     *
     * @param particles list of particles
     * @return map from each particle's id to the set of neighbor particle ids
     */
    public Map<Integer, Set<Integer>> findNeighbors(List<Particle> particles) {
        // Initialize result map
        Map<Integer, Set<Integer>> neighbors = new HashMap<>();
        for (Particle p : particles) {
            neighbors.put(p.getId(), new HashSet<>());
        }

        // Build the grid: each cell holds the list of particles in it
        List<List<Integer>> grid = buildGrid(particles);

        double cellSize = L / M;

        // Iterate over each cell
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < M; col++) {
                List<Integer> cellParticles = grid.get(cellIndex(row, col));

                // Check pares dentro de la misma celda (usar PBC si corresponde)
                checkPairsInList(
                        cellParticles,
                        cellParticles,
                        particles,
                        neighbors,
                        periodicBoundary,
                        true);

                // Check against neighbor cells (only "forward" neighbors to avoid duplicates)
                // We use the standard CIM 5-cell neighborhood (current + 4 forward neighbors)
                int[][] neighborOffsets = {
                        {0,  1},   // right
                        {1, -1},   // bottom-left
                        {1,  0},   // bottom
                        {1,  1},   // bottom-right
                };

                for (int[] offset : neighborOffsets) {
                    int nRow = row + offset[0];
                    int nCol = col + offset[1];

                    if (periodicBoundary) {
                        // Wrap around using modulo
                        nRow = Math.floorMod(nRow, M);
                        nCol = Math.floorMod(nCol, M);
                        List<Integer> neighborParticles = grid.get(cellIndex(nRow, nCol));
                        checkPairsInList(cellParticles, neighborParticles, particles, neighbors, true, false);
                    } else {
                        // Skip cells outside the grid
                        if (nRow < 0 || nRow >= M || nCol < 0 || nCol >= M) continue;
                        List<Integer> neighborParticles = grid.get(cellIndex(nRow, nCol));
                        checkPairsInList(cellParticles, neighborParticles, particles, neighbors, false, false);
                    }
                }
            }
        }

        return neighbors;
    }

    /**
     * Assigns each particle to its corresponding cell.
     *
     * @return flat list of size M*M, each entry is the list of particle ids in that cell
     */
    private List<List<Integer>> buildGrid(List<Particle> particles) {
        List<List<Integer>> grid = new ArrayList<>(M * M);
        for (int i = 0; i < M * M; i++) {
            grid.add(new ArrayList<>());
        }

        double cellSize = L / M;

        for (Particle p : particles) {
            int col = (int) (p.getX() / cellSize);
            int row = (int) (p.getY() / cellSize);

            // Clamp to valid range (handles edge case where position == L)
            col = Math.min(col, M - 1);
            row = Math.min(row, M - 1);

            grid.get(cellIndex(row, col)).add(p.getId());
        }

        return grid;
    }

    /**
     * Checks all pairs between listA and listB and adds neighbors if within rc.
     * If sameList is true, avoids checking a particle against itself.
     */
    private void checkPairsInList(
            List<Integer> listA,
            List<Integer> listB,
            List<Particle> particles,
            Map<Integer, Set<Integer>> neighbors,
            boolean usePBC,
            boolean sameCell) {

        for (int idA : listA) {
            for (int idB : listB) {
                if (sameCell && idA >= idB) {
                    continue; // evita pares repetidos dentro de la misma celda
                }

                Particle pA = particles.get(idA);
                Particle pB = particles.get(idB);

                double dist = usePBC
                        ? pA.distanceToPBC(pB, L)
                        : pA.distanceTo(pB);

                if (dist <= rc) {
                    neighbors.get(idA).add(idB);
                    neighbors.get(idB).add(idA);
                }
            }
        }
    }

    /**
     * Converts (row, col) to a flat index for the grid list.
     */
    private int cellIndex(int row, int col) {
        return row * M + col;
    }

    /**
     * Suggested optimal M for a given particle density.
     * Derived from L/M > rc + 2*rMax  =>  M < L / (rc + 2*rMax)
     *
     * @param L    side length
     * @param rc   interaction radius
     * @param rMax maximum particle radius
     * @return optimal M value
     */
    public static int optimalM(double L, double rc, double rMax) {
        return Math.max(1, (int) Math.floor(L / (rc + 2 * rMax)));
    }
}
