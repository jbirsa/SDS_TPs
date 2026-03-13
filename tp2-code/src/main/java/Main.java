import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {

        boolean leader = true;
        int leaderType = 2;

        double eta = 0.2;

        int steps = 5000;

        if (args.length >= 1) {
            leader = Boolean.parseBoolean(args[0]);
        }

        if (args.length >= 2) {
            leaderType = Integer.parseInt(args[1]);
        }

        if (args.length >= 3) {
            eta = Double.parseDouble(args[2]);
        }

        // parámetros Vicsek

        double L = 10;
        double rho = 1;
        int N = (int)(rho * L * L);
        double v = 0.03;
        double rc = 1.0;
        double dt = 1.0;

        int M = CellIndexMethod.optimalM(L, rc, 0);

        List<Particle> particles =
                Generator.generate(
                        N,
                        L,
                        v,
                        leader,
                        leaderType
                );

        CellIndexMethod cim =
                new CellIndexMethod(
                        L,
                        M,
                        rc
                );

        VicsekModel model =
                new VicsekModel(
                        particles,
                        cim,
                        eta,
                        dt,
                        L
                );

        for (int i = 0; i < steps; i++) {

            model.step();

            OutputWriter.writeFrame(
                    i,
                    model.getParticles()
            );
        }
    }
}