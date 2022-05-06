import React from 'react';
import 'react-toastify/dist/ReactToastify.css';
import PageLayout from '../../components/layout/PageLayout';
import HeaderView from './components/HeaderView';
import InstallView from './components/InstallView';
import styles from './styles.module.css';


const LandingPage: React.FC = () => {
  const getStartedRef = React.useRef<HTMLDivElement>();
  return (
    <PageLayout>
      <main className={styles.Container}>
        <HeaderView getStartedRef={getStartedRef} />
        <InstallView getStartedRef={getStartedRef} />
      </main>
    </PageLayout>
  );
};

export default LandingPage;
