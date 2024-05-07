import sys
#
# sys.path.append("../..")

from sandbox.core import SandBox, RawStateWrapper, Config, ObstacleMapGenerator, USVMission, \
    EmptyMapGenerator, BasicRenderer
import cv2


def main():
    config = Config('./test.json')
    sandbox = SandBox(config, EmptyMapGenerator, USVMission, RawStateWrapper, BasicRenderer)
    while True:
        sandbox.reset()
        while True:
            actions = sandbox.sample(zero=True)
            k = cv2.waitKey(0)
            if not k == -1:
                k = chr(k)
                if k == 'a':
                    actions[0][1] = -3
                elif k == 'd':
                    actions[0][1] = 3
                elif k == 's':
                    pass
                elif k == 'w':
                    actions[0][0] = 1
                elif k == 'x':
                    actions[0][0] = -1
                elif k == 'z':
                    actions[0][1] = -3
                    actions[0][0] = -1
                elif k == 'q':
                    actions[0][0] = 1
                    actions[0][1] = -3
                elif k == 'e':
                    actions[0][0] = 1
                    actions[0][1] = 3
                elif k == 'c':
                    actions[0][1] = 3
                    actions[0][0] = -1
            print('pressed key: {}'.format(k))
            print('current actions: {}'.format(actions))
            observation, reward, termination, truncation, info = sandbox.step(actions)
            print('env observation: ', observation)
            print('env reward: ', reward)
            print('env termination: ', termination)
            print('env truncation: ', truncation)
            sandbox.render()
            if termination or truncation:
                break


if __name__ == '__main__':
    main()
