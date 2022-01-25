import React, { useState } from 'react';
import styles from './styles.module.css';
import clsx from 'clsx';
import RocketSVG from '../../../../../../../static/img/pages/landing/rocket.svg';

type Props = { className?: string };

const Rocket: React.FC<Props> = (props) => {
  const { className } = props;

  return (
    <div className={clsx(styles.Container, className)}>
      <div className={styles.ImageContainer}>
        <RocketSVG className={className}/>
      </div>
    </div>
  );
};

export default Rocket;
