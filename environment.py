from io import BytesIO
from PIL import Image
import numpy as np
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


class KeyMgr:
    def __init__(self, driver: Chrome):
        self._space = ActionChains(driver)
        self._space.send_keys(Keys.SPACE)
        self._left = ActionChains(driver)
        self._left.send_keys(Keys.LEFT)
        self._right = ActionChains(driver)
        self._right.send_keys(Keys.RIGHT)

    def space(self):
        self._space.perform()

    def left(self):
        self._left.perform()

    def right(self):
        self._right.perform()


class LumberJackEnv:
    def __init__(self, game_url='https://tbot.xyz/lumber/'):
        self.driver = Chrome('chromedriver.exe')
        self.driver.set_window_size(width=515, height=860)
        self.driver.delete_all_cookies()
        self.game_url = game_url
        self.key = KeyMgr(self.driver)

    @property
    def n_action(self):
        return 2

    def reset(self):
        self.driver.get(self.game_url)
        self.driver.implicitly_wait(2)
        self.key.space()
        return self.get_image()

    def step(self, action):
        if action == 0:
            self.key.left()
        else:
            self.key.right()
        img = self.get_image()
        html = self.driver.page_source
        if html[606] == 'r':
            is_done = True
            reward = -10
        elif html[606] == 'g':
            is_done = False
            reward = 1
        else:
            raise Exception(f"Result Parsing Error, (606: {html[606]})[{html[587:620]}]")
        return img, reward, is_done

    def get_image(self):
        data = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(data)).convert("L")
        img = img.resize((80, 140))
        img = np.array(img)  # (80, 140)
        return img
