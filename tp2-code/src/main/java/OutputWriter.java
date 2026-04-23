import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

public class OutputWriter {

    // escribe en REPO_NUEVO/tp2-output/output.xyz sin importar desde dónde se ejecute
    private static final Path OUTPUT_PATH = resolveOutputPath();
    private static final Path ANALYSIS_PATH = resolveAnalysisPath(); // Nuevo archivo para análisis
    private static boolean initialized = false;
    private static BufferedWriter writer = null;

    private static void ensureOutputFile() throws IOException {
        if (!initialized) {
            Files.createDirectories(OUTPUT_PATH.getParent());
            Files.deleteIfExists(OUTPUT_PATH);
            writer = new BufferedWriter(new FileWriter(OUTPUT_PATH.toFile(), true));
        }
    }

    public static void writeFrame(int step, List<Particle> particles,
                                  double L, double rho, double v, double rc, double eta, int steps, int leaderType) {
        try {
            ensureOutputFile();
            if (!initialized) {
                writer.write(String.format(Locale.US,
                        "%d %.5f %.5f %.5f %.5f %.5f %d %d%n",
                        particles.size(), L, rho, v, rc, eta, steps, leaderType
                ));
                writer.newLine();
                initialized = true;
            }
        } catch (IOException e) {
            throw new RuntimeException("No se pudo preparar el archivo de salida: " + OUTPUT_PATH, e);
        }

        try {

            writer.write("step " + step);
            writer.newLine();

            for (Particle p : particles) {
                writer.write(String.format(Locale.US,
                        "%d %.5f %.5f %.5f %.5f %d%n",
                        p.getId(),
                        p.getX(),
                        p.getY(),
                        p.getVx(),
                        p.getVy(),
                        (p instanceof Leader) ? 1 : 0
                ));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeAnalysisData(double eta, int runId, double vaPromedioEstacionario) {
        try {
            Files.createDirectories(ANALYSIS_PATH.getParent());

            boolean writeHeader = !Files.exists(ANALYSIS_PATH) || Files.size(ANALYSIS_PATH) == 0;

            try (BufferedWriter analysisWriter = new BufferedWriter(new FileWriter(ANALYSIS_PATH.toFile(), true))) {

                // Header solo la primera vez
                if (writeHeader) {
                    analysisWriter.write("eta;run;va_mean\n");
                }

                // Datos
                analysisWriter.write(String.format(
                        Locale.US,
                        "%.8f;%d;%.8f%n",
                        eta,
                        runId,
                        vaPromedioEstacionario
                ));
            }

        } catch (IOException e) {
            throw new RuntimeException("No se pudo escribir en el archivo de análisis: " + ANALYSIS_PATH, e);
        }
    }

    public static void close() {
        if (writer != null) {
            try {
                writer.flush();
                writer.close();
            } catch (IOException ignored) {
            }
        }
    }

    private static Path resolveOutputPath() {
        Path cwd = Path.of("").toAbsolutePath();

        // Caso 1: se ejecuta desde tp2-code
        if (cwd.getFileName().toString().equals("tp2-code")) {
            return cwd.getParent().resolve("tp2-output").resolve("output.xyz");
        }
        // Caso 2: se ejecuta desde REPO_NUEVO
        if (cwd.getFileName().toString().equals("REPO_NUEVO")) {
            return cwd.resolve("tp2-output").resolve("output.xyz");
        }
        // Caso 3: se ejecuta desde SDS (un nivel arriba)
        if (cwd.getFileName().toString().equals("SDS")) {
            return cwd.resolve("REPO_NUEVO").resolve("tp2-output").resolve("output.xyz");
        }
        // Fallback: crear en cwd/tp2-output
        return cwd.resolve("tp2-output").resolve("output.xyz");
    }

    private static Path resolveAnalysisPath() {
        Path cwd = Path.of("").toAbsolutePath();

        // Caso 1: se ejecuta desde tp2-code
        if (cwd.getFileName().toString().equals("tp2-code")) {
            return cwd.getParent().resolve("tp2-output").resolve("analysis.csv");
        }
        // Caso 2: se ejecuta desde REPO_NUEVO
        if (cwd.getFileName().toString().equals("REPO_NUEVO")) {
            return cwd.resolve("tp2-output").resolve("analysis.csv");
        }
        // Caso 3: se ejecuta desde SDS (un nivel arriba)
        if (cwd.getFileName().toString().equals("SDS")) {
            return cwd.resolve("REPO_NUEVO").resolve("tp2-output").resolve("analysis.csv");
        }
        // Fallback: crear en cwd/tp2-output
        return cwd.resolve("tp2-output").resolve("analysis.csv");
    }

//    private static Path resolveOutputPath() {
//        return Path.of("tp2-output", "output.xyz").toAbsolutePath();
//    }
//
//    private static Path resolveAnalysisPath() {
//        return Path.of("tp2-output", "analysis.csv").toAbsolutePath();
//    }

}
