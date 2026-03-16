import java.util.*;

public class VicsekModel {

    private List<Particle> particles;
    private CellIndexMethod cim;

    private double eta;
    private double dt;
    private double L;

    public VicsekModel(List<Particle> particles, CellIndexMethod cim, double eta, double dt, double L) {
        this.particles = particles;
        this.cim = cim;
        this.eta = eta;
        this.dt = dt;
        this.L = L;
    }

    public void step() {
        Map<Integer, Set<Integer>> neighbors = cim.findNeighbors(particles);

        double[] newTheta = new double[particles.size()];

        for (Particle p : particles) {
            int id = p.getId();

            double sumX = p.getVx();
            double sumY = p.getVy();

            for (int nId : neighbors.get(id)) {
                Particle n = particles.get(nId);
                sumX += n.getVx();
                sumY += n.getVy();
            }

            double avgTheta = Math.atan2(sumY, sumX);
            double noise = eta * (Math.random() - 0.5);
            newTheta[id] = avgTheta + noise;
        }

        for (Particle p : particles) {
            p.setTheta(newTheta[p.getId()]);
            p.move(dt, L);
        }
    }

    public List<Particle> getParticles() {
        return particles;
    }
}