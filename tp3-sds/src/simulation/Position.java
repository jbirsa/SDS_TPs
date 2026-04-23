package simulation;

public class Position {
    public final double x;
    public final double y;

    public Position(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double distanceTo(Position other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public String toString() {
        return String.format("%.4f %.4f", x, y);
    }
}
