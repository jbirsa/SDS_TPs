import java.util.*;

public class Generator {

    public static List<Particle> generate(
            int N,
            double L,
            double v,
            boolean leader,
            int leaderType
    ) {

        Random rand = new Random();

        List<Particle> particles = new ArrayList<>();

        for (int i = 0; i < N; i++) {

            double x = rand.nextDouble() * L;
            double y = rand.nextDouble() * L;

            double theta = rand.nextDouble() * 2 * Math.PI;

            particles.add(
                    new Particle(i, x, y, theta, v)
            );
        }

        if (leader) {

            int id = N;

            if (leaderType == 2) {

                particles.add(
                        new CircularLeader(
                                id,
                                L / 2,
                                L / 2,
                                2,
                                v
                        )
                );

            } else if (leaderType == 1) {

                particles.add(
                        new Leader(
                                id,
                                L / 2,
                                L / 2,
                                0,
                                v
                        )
                );
            }
        }

        return particles;
    }
}