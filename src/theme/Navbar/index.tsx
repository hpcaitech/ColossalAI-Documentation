// File got generated with 'yarn run swizzle @docusaurus/theme-classic Navbar --danger'

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import BrowserOnly from '@docusaurus/BrowserOnly';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useThemeContext from '@theme/hooks/useThemeContext';
import IconMenu from '@theme/IconMenu';
import Logo from '@theme/Logo';
import NavbarItem from '@theme/NavbarItem';
import SearchBar from '@theme/SearchBar';
import Toggle from '@theme/Toggle';
import clsx from 'clsx';
import React, { useCallback } from 'react';
import { useHistory } from 'react-router-dom';
import NavbarMobileSidebar from './components/NavbarMobileSidebar';
import Progressbar from './components/Progressbar';
import QuickSocialLinksView from './components/QuickSocialLinksView';
import {
  splitNavItemsByPosition,
  useMobileSidebar,
  useNavbarItems
} from './controller';
import styles from './styles.module.css';

const Navbar: React.FC = () => {
  const { siteConfig } = useDocusaurusContext();
  const items = useNavbarItems();
  const { leftItems, rightItems } = splitNavItemsByPosition(items);
  const { isDarkTheme, setLightTheme, setDarkTheme } = useThemeContext();
  const history = useHistory();
  const mobileSidebar = useMobileSidebar();

  const onThemeToggleChange = useCallback(
    (e) => {
      if (e.target.checked) {
        setDarkTheme();
      } else {
        setLightTheme();
      }
    },
    [setLightTheme, setDarkTheme]
  );


  return (
    <nav
      className={clsx('navbar', 'navbar--fixed-top', {
        'navbar-sidebar--show': mobileSidebar.shown,
      })}>
      {/* Navbar */}
      <div className={clsx('navbar__inner', styles.InnerContainer)}>
        <div className="navbar__items">
          <Logo
            className="navbar__brand"
            imageClassName="navbar__logo"
            titleClassName="navbar__title"
          />
          <a
            className={clsx('navbar__brand', styles.BrandText)}
            onClick={() => history.push(siteConfig.baseUrl)}>
            {siteConfig.title}
          </a>
          {leftItems.map((item, i) => (
            <NavbarItem {...item} key={i} />
          ))}
        </div>
        <div className="navbar__items navbar__items--right">
          {rightItems.map((item, i) => (
            <NavbarItem {...item} key={i} />
          ))}
          <QuickSocialLinksView className={styles.displayOnlyInLargeViewport} />
          <Toggle
            aria-label="Dark mode toggle"
            checked={isDarkTheme}
            onChange={onThemeToggleChange}
          />
          <SearchBar />
        </div>
        <BrowserOnly>{() => <Progressbar />}</BrowserOnly>
      </div>

      {/* Donut */}
      {items != null && items.length !== 0 && (
        <div
          aria-label="Navigation bar toggle"
          className="navbar__toggle"
          role="button"
          tabIndex={0}
          onClick={mobileSidebar.toggle}
          onKeyDown={mobileSidebar.toggle}>
          <IconMenu />
        </div>
      )}

      {/* Mobile Sidebar Backdrop */}
      <div
        role="presentation"
        className="navbar-sidebar__backdrop"
        onClick={mobileSidebar.toggle}
      />

      {/* Mobile Sidebar */}
      {mobileSidebar.shouldRender && (
        <NavbarMobileSidebar
          sidebarShown={mobileSidebar.shown}
          toggleSidebar={mobileSidebar.toggle}
        />
      )}
    </nav>
  );
};

export default Navbar;
