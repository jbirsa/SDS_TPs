public class Leader extends Particle {

    public Leader(int id, double x, double y, double theta, double v) {
        super(id, x, y, theta, v);
    }

    @Override
    public void setTheta(double theta) {
        // el líder no cambia dirección
    }
}