package org.example;

public class Particle {
    private final int id;
    private final double x;
    private final double y;
    private final double radius;

    public Particle(int id, double x, double y, double radius) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.radius = radius;
    }

    /**
     * Edge-to-edge distance between two particles (no PBC).
     */
    public double distanceTo(Particle other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy) - this.radius - other.radius;
    }

    /**
     * Edge-to-edge distance with periodic boundary conditions.
     * Takes the shortest path across any boundary.
     */
    public double distanceToPBC(Particle other, double L) {
        double dx = Math.abs(this.x - other.x);
        double dy = Math.abs(this.y - other.y);
        // Wrap around if shorter path goes through boundary
        if (dx > L / 2) dx = L - dx;
        if (dy > L / 2) dy = L - dy;
        return Math.sqrt(dx * dx + dy * dy) - this.radius - other.radius;
    }

    public int getId()      { return id; }
    public double getX()    { return x; }
    public double getY()    { return y; }
    public double getRadius() { return radius; }

    @Override
    public String toString() {
        return String.format("Particle{id=%d, x=%.3f, y=%.3f, r=%.3f}", id, x, y, radius);
    }
}
