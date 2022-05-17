import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import React from 'react'
import styles from './styles.module.css'

type RadioGroupProps = {
    groupName: string,
    values: string[],
    selectedValue: string
    onClick: (value: string) => void,
}


const RadioGroup: React.FC<RadioGroupProps> = ({ groupName, values, selectedValue, onClick }) => {

    return (
        <div className={styles.RadioGroup}>
            {groupName}
            <ButtonGroup variant="outlined" aria-label="outlined button group" className={styles.Radio}>
                {values.map((value, idx) => <Button key={idx} variant={selectedValue == value ? "contained" : "outlined"} onClick={(e) => onClick(value)}>{value}</Button>)}
            </ButtonGroup>
        </div>
    );
};

export default RadioGroup;