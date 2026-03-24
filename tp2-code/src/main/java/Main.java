import java.util.*;

public class Main {

    private static final int STATIONARY_REPEAT_COUNT = 5;
    private static final double STATIONARY_TOLERANCE = 1e-3;
    private static final double STATIONARY_START_FRACTION = 0.4;

    public static void main(String[] args) throws Exception {
        double L = 10; // lado del cuadrado
        double rho = 4; // densidad
        int N = (int)(rho * L * L); // cantidad de particulas
        double v = 0.03; // velocidad de las particulas
        double rc = 1.0; // radio de interacción
        double dt = 1.0; // paso temporal
        double eta = 0.1; // intensidad del ruido
        int steps = 1000; // cantidad de pasos a simular
        int outputEvery = 1; // guardar cada cuantos pasos

        int leaderType = 1; // tipo de líder (0 sin lider, 1 lider con direccion fija, 2 lider con direccion circular)

        if (args.length >= 1) {
            leaderType = Integer.parseInt(args[0]);
        }

        if (args.length >= 2) {
            eta = Double.parseDouble(args[1]);
        }

        if (args.length >= 3) {
            steps = Integer.parseInt(args[2]);
        }

        if (args.length >= 4) {
            outputEvery = Integer.parseInt(args[3]);
        }

        int M = CellIndexMethod.optimalM(L, rc, 0);
        long seed = System.nanoTime();
        List<Particle> particles = Generator.generate(N, L, v, leaderType, seed);
        CellIndexMethod cim = new CellIndexMethod(L, M, rc);
        VicsekModel model = new VicsekModel(particles, cim, eta, dt, L);

        List<Double> polarizationByStep = new ArrayList<>(steps);
        for (int i = 0; i < steps; i++) {
            model.step();

            double vaCurrent = calculatePolarization(model.getParticles());
            polarizationByStep.add(vaCurrent);

            if (i % outputEvery == 0) {
                OutputWriter.writeFrame(i, model.getParticles(), L, rho, v, rc, eta, steps, leaderType);
            }
        }

        int stationaryStartIndex = detectStationaryStartIndex(polarizationByStep, steps);
        if (stationaryStartIndex < 0) {
            stationaryStartIndex = 0;
        }
        if (stationaryStartIndex >= polarizationByStep.size()) {
            stationaryStartIndex = Math.max(0, polarizationByStep.size() - 1);
        }

        List<Double> vaValues = polarizationByStep.subList(stationaryStartIndex, polarizationByStep.size());

        double vaPromedioEstacionario = vaValues.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);

        int runId = 0;
        if (args.length >= 5) {
            runId = Integer.parseInt(args[4]);
        }

        OutputWriter.writeAnalysisData(eta, runId, vaPromedioEstacionario);

        OutputWriter.close();
    }

    private static double calculatePolarization(List<Particle> particles) {
        double sumVx = 0;
        double sumVy = 0;

        for (Particle p : particles) {
            sumVx += p.getVx();
            sumVy += p.getVy();
        }

        double magnitude = Math.sqrt(sumVx * sumVx + sumVy * sumVy);
        return magnitude / particles.size();
    }

    private static int detectStationaryStartIndex(List<Double> values, int totalSteps) {
        Integer startByWindow = detectStationaryStartIndexByWindow(
                values,
                STATIONARY_REPEAT_COUNT,
                STATIONARY_TOLERANCE
        );
        if (startByWindow != null) {
            return startByWindow;
        }
        return detectStationaryStartIndexByFraction(totalSteps, STATIONARY_START_FRACTION);
    }

    private static Integer detectStationaryStartIndexByWindow(
            List<Double> values,
            int repeatCount,
            double tolerance
    ) {
        if (repeatCount <= 0 || values.isEmpty() || values.size() < repeatCount) {
            return null;
        }
        if (repeatCount == 1) {
            return 0;
        }

        for (int start = 0; start <= values.size() - repeatCount; start++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            for (int i = start; i < start + repeatCount; i++) {
                double v = values.get(i);
                if (v < min) {
                    min = v;
                }
                if (v > max) {
                    max = v;
                }
            }

            if ((max - min) <= tolerance) {
                return start;
            }
        }

        return null;
    }

    private static int detectStationaryStartIndexByFraction(int totalSteps, double startFraction) {
        if (totalSteps <= 0) {
            return 0;
        }
        int threshold = (int) (startFraction * totalSteps);
        int start = threshold + 1;
        if (start < 0) {
            start = 0;
        }
        if (start >= totalSteps) {
            return totalSteps - 1;
        }
        return start;
    }
}
