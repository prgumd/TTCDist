###############################################################################
#
# File: ttc_depth_plot_live.py
# Available under MIT license
#
# Plot states published by ttc_depth.py over a ZMQ socket
#
# History:
# 08-30-21 - Levi Burner - Created File
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import struct
import threading

import numpy as np
import zmq

import matplotlib.pyplot as plt
import matplotlib.animation as animation

port = '5556'
STOPPED = False

sensors = {}
sensors[b'R_fc_to_c'] = []
sensors[b'p'] = []
sensors[b'ttc_inv'] = []
sensors[b'ttc_inv_gt'] = []
sensors[b'pose_hat'] = []
sensors[b'phi_pose_hat'] = []
sensors[b'ground_truth_pose'] = []
sensors[b'accel_meas_c'] = []
sensors[b'gyro'] = []
sensors[b'accel_z_hat'] = []
sensors[b'phi_accel_z_hat'] = []

def zmq_receive_thread(port):
    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    print('Collecting updates from server...')
    socket.connect ("tcp://localhost:{}".format(port))

    topicfilter = "ttc_depth".encode('ascii')
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

    while not STOPPED:
        topic = socket.recv()
        base_name, sensor = topic.split(b'/')

        time_bytes = socket.recv()
        t = struct.unpack('d', time_bytes)

        md = socket.recv_json()
        msg = socket.recv()
        buf = memoryview(msg)
        x = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])

        try:
            sensors[sensor].append((t, x))
        except KeyError as e:
            print('Unrecognized sensor: {}'.format(sensor))

# Plotting based on: https://learn.sparkfun.com/tutorials/graph-sensor-data-with-python-and-matplotlib/speeding-up-the-plot-animation
def ttc_depth_plot_live_process():
    x_len = 300
    xs = list(range(0, x_len))
    fig = plt.figure()

    ROWS = 4
    COLS = 3

    Z_LIM = (-4.0, 0.5)

    F_LIM = (-4.0, 4.0)

    # Setup pose plots
    ly1 = [0] * x_len
    ly2 = [0] * x_len
    ly3 = [0] * x_len
    ly4 = [0] * x_len
    ly5 = [0] * x_len
    ly6 = [0] * x_len


    ax = fig.add_subplot(ROWS, COLS, 1)
    ax.set_ylim([-1, 1])
    lline1, = ax.plot(xs, ly1)
    lline4, = ax.plot(xs, ly4)
    llllllllllly1 = [0] * x_len
    lllllllllllline1, = ax.plot(xs, llllllllllly1)
    plt.ylabel('X (m)')
    plt.legend(['x_hat', 'x_gt', 'phi_x_hat'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 2)
    ax.set_ylim([-1, 1])
    lline2, = ax.plot(xs, ly2)
    lline5, = ax.plot(xs, ly5)
    llllllllllly2 = [0] * x_len
    lllllllllllline2, = ax.plot(xs, llllllllllly2)
    plt.ylabel('Y (m)')
    plt.legend(['y_hat', 'y_gt', 'phi_y_hat'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 3)
    ax.set_ylim(Z_LIM)
    lline3, = ax.plot(xs, ly3)
    lline6, = ax.plot(xs, ly6)
    llllllllllly3 = [0] * x_len
    lllllllllllline3, = ax.plot(xs, llllllllllly3)
    plt.ylabel('Z (m)')
    plt.legend(['z_hat', 'z_gt', 'phi_z_hat'])
    plt.grid()

    # Setup Orientation plot
    ax = fig.add_subplot(ROWS, COLS, 4, projection='3d')
    axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    lines = []
    for a in axis:
        line_data = np.array([[0, 0, 0], [a[0], a[1], a[2]]]).transpose()
        line = ax.plot(line_data[0, :], line_data[1, :], line_data[2, :])[0]
        lines.append(line)

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_title('Integrated Orientation')
    plt.legend(['x', 'y', 'z'])

    ax = fig.add_subplot(ROWS, COLS, 5)
    ax.set_ylim([-10, 10])
    llly1 = [0] * x_len
    llly2 = [0] * x_len
    llly3 = [0] * x_len
    lllline1, = ax.plot(xs, llly1)
    lllline2, = ax.plot(xs, llly2)
    lllline3, = ax.plot(xs, llly3)
    plt.ylabel('accel (m/s^2)')
    plt.legend(['x','y','z'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 6)
    ax.set_ylim([-np.pi, np.pi])
    lllly1 = [0] * x_len
    lllly2 = [0] * x_len
    lllly3 = [0] * x_len
    llllline1, = ax.plot(xs, lllly1)
    llllline2, = ax.plot(xs, lllly2)
    llllline3, = ax.plot(xs, lllly3)
    plt.ylabel('gyro (rad/s)')
    plt.legend(['x','y','z'])
    plt.grid()


    ax = fig.add_subplot(ROWS, COLS, 7)
    ax.set_ylim(F_LIM)
    llllly1 = [0] * x_len
    llllly2 = [0] * x_len
    lllllline1, = ax.plot(xs, llllly1)
    lllllline2, = ax.plot(xs, llllly2)
    plt.ylabel('depth scaled velocity 1/s')
    plt.legend(['dot x / z', 'dot x / z gt'])
    plt.grid()


    ax = fig.add_subplot(ROWS, COLS, 8)
    ax.set_ylim(F_LIM)
    lllllly1 = [0] * x_len
    lllllly2 = [0] * x_len
    llllllline1, = ax.plot(xs, lllllly1)
    llllllline2, = ax.plot(xs, lllllly2)
    plt.ylabel('depth scaled velocity 1/s')
    plt.legend(['dot y / z', 'dot y / z gt'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 9)
    ax.set_ylim(F_LIM)
    llllllly1 = [0] * x_len
    llllllly2 = [0] * x_len
    lllllllline1, = ax.plot(xs, llllllly1)
    lllllllline2, = ax.plot(xs, llllllly2)
    plt.ylabel('depth scaled velocity 1/s')
    plt.legend(['dot z / z', 'dot z / z gt'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 10)
    ax.set_ylim(Z_LIM)
    lllllllly1 = [0] * x_len
    llllllllly1 = [0] * x_len
    lllllllllly1 = [0] * x_len
    lllllllllllly1 = [0] * x_len
    llllllllllllly1 = [0] * x_len
    llllllllline1, = ax.plot(xs, lllllllly1)
    lllllllllline1, = ax.plot(xs, llllllllly1)
    llllllllllline1, = ax.plot(xs, lllllllllly1)
    llllllllllllline1, = ax.plot(xs, lllllllllllly1)
    lllllllllllllline1, = ax.plot(xs, llllllllllllly1)
    plt.ylabel('X (m)')
    plt.legend(['accel_x_gt', 'x_hat', 'x_gt', 'phi_x_hat', 'p_accel_x_gt'])
    plt.grid()


    ax = fig.add_subplot(ROWS, COLS, 11)
    ax.set_ylim(Z_LIM)
    lllllllly2 = [0] * x_len
    llllllllly2 = [0] * x_len
    lllllllllly2 = [0] * x_len
    lllllllllllly2 = [0] * x_len
    llllllllllllly2 = [0] * x_len
    llllllllline2, = ax.plot(xs, lllllllly2)
    lllllllllline2, = ax.plot(xs, llllllllly2)
    llllllllllline2, = ax.plot(xs, lllllllllly2)
    llllllllllllline2, = ax.plot(xs, lllllllllllly2)
    lllllllllllllline2, = ax.plot(xs, llllllllllllly2)
    plt.ylabel('Y (m)')
    plt.legend(['accel_y_gt', 'y_hat', 'y_gt', 'phi_y_hat', 'p_accel_y_gt'])
    plt.grid()

    ax = fig.add_subplot(ROWS, COLS, 12)
    ax.set_ylim(Z_LIM)
    lllllllly3 = [0] * x_len
    llllllllly3 = [0] * x_len
    lllllllllly3 = [0] * x_len
    lllllllllllly3 = [0] * x_len
    llllllllllllly3 = [0] * x_len
    llllllllline3, = ax.plot(xs, lllllllly3)
    lllllllllline3, = ax.plot(xs, llllllllly3)
    llllllllllline3, = ax.plot(xs, lllllllllly3)
    llllllllllllline3, = ax.plot(xs, lllllllllllly3)
    lllllllllllllline3, = ax.plot(xs, llllllllllllly3)
    plt.ylabel('Z (m)')
    plt.legend(['accel_z_gt', 'z_hat', 'z_gt', 'phi_z_hat', 'p_accel_z_gt'])
    plt.grid()

    # This function is called periodically from FuncAnimation
    def animate(i,
                sensors,
                ly1, ly2, ly3, ly4, ly5, ly6,
                lines,
                llly1, llly2, llly3,
                lllly1, lllly2, lllly3,
                llllly1, llllly2,
                lllllly1, lllllly2,
                llllllly1, llllllly2,
                lllllllly1, lllllllly2, lllllllly3,
                llllllllly1, llllllllly2, llllllllly3,
                lllllllllly1, lllllllllly2, lllllllllly3,
                llllllllllly1, llllllllllly2, llllllllllly3,
                lllllllllllly1, lllllllllllly2, lllllllllllly3,
                llllllllllllly1, llllllllllllly2, llllllllllllly3):
        # Plot linear acceleration
        if len(sensors[b'pose_hat']) > 0:
            ly1.append(sensors[b'pose_hat'][-1][1][0])
            ly2.append(sensors[b'pose_hat'][-1][1][1])
            ly3.append(sensors[b'pose_hat'][-1][1][2])

            ly1 = ly1[-x_len:]
            ly2 = ly2[-x_len:]
            ly3 = ly3[-x_len:]

            lline1.set_ydata(ly1)
            lline2.set_ydata(ly2)
            lline3.set_ydata(ly3)

            # On 4th row 3 plots
            llllllllly1.append(sensors[b'pose_hat'][-1][1][2])
            llllllllly2.append(sensors[b'pose_hat'][-1][1][2])
            llllllllly3.append(sensors[b'pose_hat'][-1][1][2])

            llllllllly1 = llllllllly1[-x_len:]
            llllllllly2 = llllllllly2[-x_len:]
            llllllllly3 = llllllllly3[-x_len:]

            lllllllllline1.set_ydata(llllllllly1)
            lllllllllline2.set_ydata(llllllllly2)
            lllllllllline3.set_ydata(llllllllly3)

        if len(sensors[b'phi_pose_hat']) > 0:
            llllllllllly1.append(sensors[b'phi_pose_hat'][-1][1][0])
            llllllllllly2.append(sensors[b'phi_pose_hat'][-1][1][1])
            llllllllllly3.append(sensors[b'phi_pose_hat'][-1][1][2])

            llllllllllly1 = llllllllllly1[-x_len:]
            llllllllllly2 = llllllllllly2[-x_len:]
            llllllllllly3 = llllllllllly3[-x_len:]

            lllllllllllline1.set_ydata(llllllllllly1)
            lllllllllllline2.set_ydata(llllllllllly2)
            lllllllllllline3.set_ydata(llllllllllly3)

            # On 4th row 3 plots
            lllllllllllly1.append(sensors[b'phi_pose_hat'][-1][1][2])
            lllllllllllly2.append(sensors[b'phi_pose_hat'][-1][1][2])
            lllllllllllly3.append(sensors[b'phi_pose_hat'][-1][1][2])

            lllllllllllly1 = lllllllllllly1[-x_len:]
            lllllllllllly2 = lllllllllllly2[-x_len:]
            lllllllllllly3 = lllllllllllly3[-x_len:]

            llllllllllllline1.set_ydata(lllllllllllly1)
            llllllllllllline2.set_ydata(lllllllllllly2)
            llllllllllllline3.set_ydata(lllllllllllly3)

        if len(sensors[b'ground_truth_pose']) > 0:
            ly4.append(sensors[b'ground_truth_pose'][-1][1][0])
            ly5.append(sensors[b'ground_truth_pose'][-1][1][1])
            ly6.append(sensors[b'ground_truth_pose'][-1][1][2])

            ly4 = ly4[-x_len:]
            ly5 = ly5[-x_len:]
            ly6 = ly6[-x_len:]

            lline4.set_ydata(ly4)
            lline5.set_ydata(ly5)
            lline6.set_ydata(ly6)

            # On 4th row 3 plots
            lllllllllly1.append(sensors[b'ground_truth_pose'][-1][1][2])
            lllllllllly2.append(sensors[b'ground_truth_pose'][-1][1][2])
            lllllllllly3.append(sensors[b'ground_truth_pose'][-1][1][2])

            lllllllllly1 = lllllllllly1[-x_len:]
            lllllllllly2 = lllllllllly2[-x_len:]
            lllllllllly3 = lllllllllly3[-x_len:]

            llllllllllline1.set_ydata(lllllllllly1)
            llllllllllline2.set_ydata(lllllllllly2)
            llllllllllline3.set_ydata(lllllllllly3)

        # Plot orientation axis
        if len(sensors[b'R_fc_to_c']) > 0:
            t, R_fc_to_c = sensors[b'R_fc_to_c'][-1]
        else:
            R_fc_to_c = np.eye(3)
        
        axis_rotated = R_fc_to_c @ axis

        for a, line in zip(axis_rotated.transpose(), lines):
            line_data = np.array([[0, 0, 0], [a[0], a[1], a[2]]]).transpose()
            line.set_data(line_data[0:2, :])
            line.set_3d_properties(line_data[2, :])

        # Plot acceleration
        if len(sensors[b'accel_meas_c']) > 0:
            llly1.append(sensors[b'accel_meas_c'][-1][1][0])
            llly2.append(sensors[b'accel_meas_c'][-1][1][1])
            llly3.append(sensors[b'accel_meas_c'][-1][1][2])
            llly1 = llly1[-x_len:]
            llly2 = llly2[-x_len:]
            llly3 = llly3[-x_len:]

            lllline1.set_ydata(llly1)
            lllline2.set_ydata(llly2)
            lllline3.set_ydata(llly3)

        # Plot gyro
        if len(sensors[b'gyro']) > 0:
            lllly1.append(sensors[b'gyro'][-1][1][0])
            lllly2.append(sensors[b'gyro'][-1][1][1])
            lllly3.append(sensors[b'gyro'][-1][1][2])
            lllly1 = lllly1[-x_len:]
            lllly2 = lllly2[-x_len:]
            lllly3 = lllly3[-x_len:]

            llllline1.set_ydata(lllly1)
            llllline2.set_ydata(lllly2)
            llllline3.set_ydata(lllly3)


        # Plot ttc_inv
        if len(sensors[b'ttc_inv']) > 0:
            llllly1.append(sensors[b'ttc_inv'][-1][1][0])
            llllly1 = llllly1[-x_len:]

            lllllly1.append(sensors[b'ttc_inv'][-1][1][1])
            lllllly1 = lllllly1[-x_len:]

            llllllly1.append(sensors[b'ttc_inv'][-1][1][2])
            llllllly1 = llllllly1[-x_len:]

            lllllline1.set_ydata(llllly1)
            llllllline1.set_ydata(lllllly1)
            lllllllline1.set_ydata(llllllly1)

        # Plot ttc_inv_gt
        if len(sensors[b'ttc_inv_gt']) > 0:
            llllly2.append(sensors[b'ttc_inv_gt'][-1][1][0])
            llllly2 = llllly2[-x_len:]

            lllllly2.append(sensors[b'ttc_inv_gt'][-1][1][1])
            lllllly2 = lllllly2[-x_len:]

            llllllly2.append(sensors[b'ttc_inv_gt'][-1][1][2])
            llllllly2 = llllllly2[-x_len:]

            lllllline2.set_ydata(llllly2)
            llllllline2.set_ydata(lllllly2)
            lllllllline2.set_ydata(llllllly2)


        # Plot accel_z_hat
        if len(sensors[b'accel_z_hat']) > 0:
            lllllllly1.append(sensors[b'accel_z_hat'][-1][1][0])
            lllllllly1 = lllllllly1[-x_len:]

            llllllllline1.set_ydata(lllllllly1)

            lllllllly2.append(sensors[b'accel_z_hat'][-1][1][1])
            lllllllly2 = lllllllly2[-x_len:]

            llllllllline2.set_ydata(lllllllly2)

            lllllllly3.append(sensors[b'accel_z_hat'][-1][1][2])
            lllllllly3 = lllllllly3[-x_len:]

            llllllllline3.set_ydata(lllllllly3)

        # Plot phi_accel_z_hat
        if len(sensors[b'phi_accel_z_hat']) > 0:
            llllllllllllly1.append(sensors[b'phi_accel_z_hat'][-1][1][0])
            llllllllllllly1 = llllllllllllly1[-x_len:]

            lllllllllllllline1.set_ydata(llllllllllllly1)

            llllllllllllly2.append(sensors[b'phi_accel_z_hat'][-1][1][1])
            llllllllllllly2 = llllllllllllly2[-x_len:]

            lllllllllllllline2.set_ydata(llllllllllllly2)

            llllllllllllly3.append(sensors[b'phi_accel_z_hat'][-1][1][2])
            llllllllllllly3 = llllllllllllly3[-x_len:]

            lllllllllllllline3.set_ydata(llllllllllllly3)

        return (lline1, lline2, lline3, lline4, lline5, lline6,
                lines[0], lines[1], lines[2],
                lllline1, lllline2, lllline3, 
                llllline1, llllline2, llllline3,
                lllllline1, lllllline2,
                llllllline1, llllllline2,
                lllllllline1, lllllllline2,
                llllllllline1, llllllllline2, llllllllline3,
                lllllllllline1, lllllllllline2, lllllllllline3,
                llllllllllline1, llllllllllline2, llllllllllline3,
                lllllllllllline1, lllllllllllline2, lllllllllllline3,
                llllllllllllline1, llllllllllllline2, llllllllllllline3,
                lllllllllllllline1, lllllllllllllline2, lllllllllllllline3)

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig,
        animate,
        fargs=(sensors,
               ly1, ly2, ly3, ly4, ly5, ly6,
               lines,
               llly1, llly2, llly3,
               lllly1, lllly2, lllly3,
               llllly1, llllly2,
               lllllly1, lllllly2,
               llllllly1, llllllly2,
               lllllllly1, lllllllly2, lllllllly3,
               llllllllly1, llllllllly2, llllllllly3,
               lllllllllly1, lllllllllly2, lllllllllly3,
               llllllllllly1, llllllllllly2, llllllllllly3,
               lllllllllllly1, lllllllllllly2, lllllllllllly3,
               llllllllllllly1, llllllllllllly2, llllllllllllly3),
        interval=10,
        blit=True)

    plt.show()
    print('ttc_depth_live_plot_process exiting')

if __name__ == '__main__':
    zmq_thread = threading.Thread(target=zmq_receive_thread, args=(port,))
    zmq_thread.start()

    ttc_depth_plot_live_process()

    STOPPED = True
    zmq_thread.join()
