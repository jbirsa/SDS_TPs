public class Particle {

    protected int id;

    protected double x;
    protected double y;

    protected double theta;

    protected double vx;
    protected double vy;

    protected double v;

    public Particle(int id, double x, double y, double theta, double v) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.theta = theta;
        this.v = v;

        this.vx = Math.cos(theta);
        this.vy = Math.sin(theta);
    }

    public void setTheta(double theta) {

        this.theta = theta;

        this.vx = Math.cos(theta);
        this.vy = Math.sin(theta);
    }

    public void move(double dt, double L) {

        x += v * vx * dt;
        y += v * vy * dt;

        x = (x % L + L) % L;
        y = (y % L + L) % L;
    }

    public double distanceTo(Particle other, double L) {

        double dx = Math.abs(x - other.x);
        double dy = Math.abs(y - other.y);

        dx = Math.min(dx, L - dx);
        dy = Math.min(dy, L - dy);

        return Math.sqrt(dx*dx + dy*dy);
    }

    public int getId() { return id; }

    public double getTheta() { return theta; }

    public double getVx() { return vx; }

    public double getVy() { return vy; }

    public double getX() { return x; }

    public double getY() { return y; }
}