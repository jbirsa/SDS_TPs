public class CircularLeader extends Leader {

    private double xc;
    private double yc;
    private double R;
    private double omega;
    private double angle;

    public CircularLeader(int id, double xc, double yc, double R, double v) {
        super(id, xc + R, yc, 0, v);
        this.xc = xc;
        this.yc = yc;
        this.R = R;
        this.omega = v / R;
        this.angle = 0;
    }

    @Override
    public void move(double dt, double L) {
        angle += omega * dt;
        x = xc + R * Math.cos(angle);
        y = yc + R * Math.sin(angle);
        theta = angle + Math.PI / 2;
        vx = Math.cos(theta);
        vy = Math.sin(theta);
    }
}