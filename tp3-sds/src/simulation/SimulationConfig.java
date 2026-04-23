package simulation;

/**
 * Immutable configuration for one simulation run.
 *
 * <p>Use {@link Builder} to construct instances:
 * <pre>
 *   SimulationConfig cfg = new SimulationConfig.Builder()
 *       .meanArrivalTime(1.0)
 *       .meanServiceTime(3.0)
 *       .numServers(5)
 *       .modality(Modality.A)
 *       .queueType(QueueType.SERPENTINE)
 *       .simulationTime(300.0)
 *       .outputFile("output.txt")
 *       .seed(42L)
 *       .build();
 * </pre>
 */
public class SimulationConfig {

    public enum Modality  { A, B }             // A = one queue per server; B = shared queue
    public enum QueueType { SERPENTINE, FREE } // guided serpentine or free-space heuristic

    public final double     roomSize;          // side of the square room in metres
    public final double     walkingSpeed;      // m/s
    public final double     meanArrivalTime;   // 1/lambda for the Poisson arrival process
    public final double     meanServiceTime;   // mean of the exponential service time
    public final int        numServers;        // k
    public final Modality   modality;
    public final QueueType  queueType;
    public final double     simulationTime;    // total simulation horizon in seconds
    public final String     outputFile;
    public final long       seed;

    private SimulationConfig(Builder b) {
        this.roomSize        = b.roomSize;
        this.walkingSpeed    = b.walkingSpeed;
        this.meanArrivalTime = b.meanArrivalTime;
        this.meanServiceTime = b.meanServiceTime;
        this.numServers      = b.numServers;
        this.modality        = b.modality;
        this.queueType       = b.queueType;
        this.simulationTime  = b.simulationTime;
        this.outputFile      = b.outputFile;
        this.seed            = b.seed;
    }

    public static class Builder {
        private double     roomSize        = 30.0;
        private double     walkingSpeed    = 1.0;
        private double     meanArrivalTime = 1.0;
        private double     meanServiceTime = 3.0;
        private int        numServers      = 5;
        private Modality   modality        = Modality.A;
        private QueueType  queueType       = QueueType.SERPENTINE;
        private double     simulationTime  = 300.0;
        private String     outputFile      = "output.txt";
        private long       seed            = System.currentTimeMillis();

        public Builder roomSize(double v)        { roomSize = v;        return this; }
        public Builder walkingSpeed(double v)    { walkingSpeed = v;    return this; }
        public Builder meanArrivalTime(double v) { meanArrivalTime = v; return this; }
        public Builder meanServiceTime(double v) { meanServiceTime = v; return this; }
        public Builder numServers(int v)         { numServers = v;      return this; }
        public Builder modality(Modality v)      { modality = v;        return this; }
        public Builder queueType(QueueType v)    { queueType = v;       return this; }
        public Builder simulationTime(double v)  { simulationTime = v;  return this; }
        public Builder outputFile(String v)      { outputFile = v;      return this; }
        public Builder seed(long v)              { seed = v;            return this; }

        public SimulationConfig build() { return new SimulationConfig(this); }
    }
}
