import java.io.*;
import java.util.*;

public class OutputWriter {

    public static void writeFrame(int step, List<Particle> particles) {

        try (PrintWriter pw = new PrintWriter(
                new FileWriter("output.xyz", true))) {

            pw.println(particles.size());
            pw.println("step " + step);

            for (Particle p : particles) {

                pw.printf(
                        "%d %.5f %.5f %.5f %.5f %f %d %d\n",
                        p.getId(),
                        p.getX(),
                        p.getY(),
                        p.getVx(),
                        p.getVy(),
                        0.2,
                        (p instanceof Leader) ? 0 : 1,
                        (p instanceof Leader) ? 1 : 0
                );
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}