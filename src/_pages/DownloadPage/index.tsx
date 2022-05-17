import React from 'react';
import 'react-toastify/dist/ReactToastify.css';
import PageLayout from '../../components/layout/PageLayout';
import InstallView from './components/InstallView';
import styles from './styles.module.css';


const DownloadPage: React.FC = () => {
  const getStartedRef = React.useRef<HTMLDivElement>();
  return (
    <PageLayout>
      <main className={styles.Container}>
        <InstallView getStartedRef={getStartedRef} />
      </main>
    </PageLayout>
  );
};

export default DownloadPage;
