import mpi.*;

public class Main {
    static final int NRA = 2000;
    static final int NCA = 2000;
    static final int NCB = 2000;
    static final int MASTER = 0;

    public static void main(String[] args) {
        performMatrixMultiplicationCollective(args);
    }

    public static void performMatrixMultiplicationCollective(String[] args) {
        MPI.Init(args);

        int currentProcess = MPI.COMM_WORLD.Rank();
        int processesCount = MPI.COMM_WORLD.Size();

        if (NRA % processesCount != 0) {
            if (currentProcess == MASTER) {
                System.out.println("Error!");
            }

            MPI.Finalize();
            return;
        }

        int rowsPerProcess = NRA / processesCount;

        Matrix matrixB = new Matrix(NCA, NCB);
        double[] arrSubA = new double[rowsPerProcess * NCA];
        double[] arrB = new double[NCA * NCB];
        double[] arrA = new double[NRA * NCA];

        long startTime = 0;

        if (currentProcess == MASTER) {
            Matrix matrixA = new Matrix(NRA, NCA);
            matrixA.fill(2);
            matrixB.fill(2);

//            matrixA.print();
//            matrixB.print();

            startTime = System.nanoTime();
            arrB = matrixB.convertToArray1D();
            arrA = matrixA.convertToArray1D();
        }

        MPI.COMM_WORLD.Bcast(arrB, 0, NCA * NCB, MPI.DOUBLE, MASTER);
        MPI.COMM_WORLD.Scatter(arrA, 0, rowsPerProcess * NCA, MPI.DOUBLE,
                arrSubA, 0, rowsPerProcess * NCA, MPI.DOUBLE, MASTER);

        matrixB = new Matrix(arrB, NCA, NCB);
        Matrix subMatrixA = new Matrix(arrSubA, rowsPerProcess, NCA);
        Matrix matrixSubC = new Matrix(rowsPerProcess, NCB);

        for (int i = 0; i < rowsPerProcess; i++) {
            for (int j = 0; j < NCB; j++) {
                for (int k = 0; k < NCA; k++) {
                    matrixSubC.setElement(i, j, matrixSubC.getElement(i, j) + subMatrixA.getElement(i, k) * matrixB.getElement(k, j));
                }
            }
        }

        double[] arrSubC = matrixSubC.convertToArray1D();
        double[] arrC = new double[NRA * NCB];

        MPI.COMM_WORLD.Gather(arrSubC, 0, rowsPerProcess * NCB, MPI.DOUBLE,
                arrC, 0, rowsPerProcess * NCB, MPI.DOUBLE, MASTER);

        if (currentProcess == MASTER) {
            Matrix matrixC = new Matrix(NRA, NCB);
            matrixC.setStripeByArr1D(arrC, 0);
            long endTime = System.nanoTime();
            System.out.println("Processes count: " + processesCount);
            System.out.println("Matrix A dimensions: " + NRA + "x" + NCA);
            System.out.println("Matrix B dimensions: " + NCA + "x" + NCB);
            System.out.println("Matrix C dimensions: " + NRA + "x" + NCB);
            System.out.println("Matrix Multiplication time (MPI Collective): " + (endTime - startTime) / 1000000000.0);
//            matrixC.print();
        }

        MPI.Finalize();
    }
}