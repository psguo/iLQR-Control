from arms.arm_python import Arm
from controllers.iLQR import iLQR
from tasks.reach import Task
from sim_and_plot import Runner

arm = Arm(dt=1e-2)
controller_class = iLQR
task = Task

control_shell, runner_pars = task(
    arm, controller_class,
    sequence=args['--sequence'], scale=args['--scale'],
    force=float(args['--force']) if args['--force'] is not None else None,
    write_to_file=bool(args['--write_to_file']))

# set up simulate and plot system
runner = Runner(dt=dt, **runner_pars)
runner.run(arm=arm, control_shell=control_shell,
           end_time=(float(args['--end_time'])
                     if args['--end_time'] is not None else None))
runner.show()
