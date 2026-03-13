import java.util.List;

public class OrderParameter {

    public static double compute(List<Particle> particles) {

        double vx = 0;
        double vy = 0;

        for (Particle p : particles) {
            vx += Math.cos(p.getTheta());
            vy += Math.sin(p.getTheta());
        }

        double norm = Math.sqrt(vx*vx + vy*vy);

        return norm / particles.size();
    }
}
