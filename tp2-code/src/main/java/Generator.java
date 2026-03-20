import java.util.*;

public class Generator {

    public static List<Particle> generate(int N, double L, double v, int leaderType, long seed) {
        Random rand = new Random(seed);
        List<Particle> particles = new ArrayList<>();

        // generar N partículas normales
        for (int i = 0; i < N; i++) {
            double x = rand.nextDouble() * L;
            double y = rand.nextDouble() * L;
            double theta = rand.nextDouble() * 2 * Math.PI;
            particles.add(new Particle(i, x, y, theta, v));
        }

        // si hay líder → reemplazar una partícula (por ejemplo la 0)
        if (leaderType == 1) {
            Particle old = particles.get(0);
            particles.set(0, new Leader(
                    old.getId(),
                    old.getX(),
                    old.getY(),
                    old.getTheta(),
                    v
            ));
        } else if (leaderType == 2) {
            // líder circular centrado en el sistema
            particles.set(0, new CircularLeader(
                    particles.get(0).getId(),
                    L / 2,
                    L / 2,
                    5,   // radio que pide la consigna
                    v
            ));
        }

        return particles;
    }
}