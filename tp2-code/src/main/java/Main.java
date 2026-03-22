import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {
        double L = 10; // lado del cuadrado
        double rho = 8; // densidad
        int N = (int)(rho * L * L); // cantidad de particulas
        double v = 0.03; // velocidad de las particulas
        double rc = 1.0; // radio de interacción
        double dt = 1.0; // paso temporal
        double eta = 3; // intensidad del ruido
        int steps = 1000; // cantidad de pasos a simular
        int outputEvery = 1; // guardar cada cuantos pasos

        int leaderType = 2; // tipo de líder (0 sin lider, 1 lider con direccion fija, 2 lider con direccion circular)

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

        List<Double> vaValues = new ArrayList<>();
        for (int i = 0; i < steps; i++) {
            model.step();
            if (i > (int)(0.4 * steps)) { // Solo considerar pasos estacionarios
                vaValues.add(calculatePolarization(model.getParticles()));
            }
            if (i % outputEvery == 0) {
                OutputWriter.writeFrame(i, model.getParticles());
            }
        }

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
}
