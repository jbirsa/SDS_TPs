import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {
        double L = 10; // lado del cuadrado
        double rho = 4; // densidad
        int N = (int)(rho * L * L); // cantidad de particulas
        double v = 0.03; // velocidad de las particulas
        double rc = 1.0; // radio de interacción
        double dt = 1.0; // paso de tiempo
        double eta = 0.1; // intensidad del ruido
        int leaderType = 2; // tipo de líder (0 sin lider, 1 lider con direccion fija, 2 lider con direccion circular)
        int steps = 5000; // cantidad de pasos a simular

        if (args.length >= 1) {
            leaderType = Integer.parseInt(args[0]);
        }

        if (args.length >= 2) {
            eta = Double.parseDouble(args[1]);
        }

        int M = CellIndexMethod.optimalM(L, rc, 0);
        List<Particle> particles = Generator.generate(N, L, v, leaderType);
        CellIndexMethod cim = new CellIndexMethod(L, M, rc);
        VicsekModel model = new VicsekModel(particles, cim, eta, dt, L);

        for (int i = 0; i < steps; i++) {
            model.step();
            OutputWriter.writeFrame(i, model.getParticles());
        }
    }
}