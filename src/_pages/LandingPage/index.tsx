import React from 'react';
import 'react-toastify/dist/ReactToastify.css';
import PageLayout from '../../components/layout/PageLayout';
import HeaderView from './components/HeaderView';
import Overview from './components/indexOverview';
import styles from './styles.module.css';


const LandingPage: React.FC = () => {
  const getStartedRef = React.useRef<HTMLDivElement>();
  return (
    <PageLayout>
      <main className={styles.Container}>
        <HeaderView getStartedRef={getStartedRef} />
          <Overview />
      </main>
    </PageLayout>
  );
};

export default LandingPage;
