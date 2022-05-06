import React, { useEffect, useState } from 'react';
import { useWindowSize } from '../../../../hooks/useWindowSize';
import styles from './styles.module.css'
import { getPipPkgVersions, getColossalaiVersions, getTorchVersions, getCudaVersions } from './pipPackages'
import RadioGroup from './components/RadioGroup';

type Props = { getStartedRef: React.RefObject<HTMLDivElement> }

const InstallView: React.FC<Props> = ({ getStartedRef }) => {
    const { windowHeight } = useWindowSize();

    const [pipPkgVersions, setPipPkgVersions] = useState(new Set())
    const [colossalaiVersions, setColossalaiVersions] = useState(new Array())
    const [torchVersions, setTorchVersions] = useState(new Array())
    const [cudaVersions, setCudaVersions] = useState(new Array())

    const [colossalaiVersion, setColossalaiVersion] = useState('')
    const [torchVersion, setTorchVersion] = useState('');
    const [cudaVersion, setCudaVersion] = useState('')

    useEffect(() => {
        async function setupPkg() {
            const pipPkgVersions = await getPipPkgVersions()
            const colossalaiVersions = getColossalaiVersions(pipPkgVersions)
            const torchVersions = getTorchVersions(pipPkgVersions)
            const cudaVersions = getCudaVersions(pipPkgVersions)

            setPipPkgVersions(pipPkgVersions)
            setColossalaiVersions(colossalaiVersions)
            setTorchVersions(torchVersions)
            setCudaVersions(cudaVersions)

            setColossalaiVersion(colossalaiVersions[colossalaiVersions.length - 1])
            setTorchVersion(torchVersions[torchVersions.length - 1])
            setCudaVersion(cudaVersions[cudaVersions.length - 1])
        }
        setupPkg()
    }, [])


    const resolveCommand = () => {
        const targetVersion = `${colossalaiVersion}+torch${torchVersion}cu${cudaVersion}`
        if (pipPkgVersions.has(targetVersion)) {
            return `pip install colossalai==${targetVersion} -f https://release.colossalai.org`
        } else {
            return 'Not available. Please install from source.'
        }
    }

    return (
        <div
            className={styles.Container}
            style={{ height: windowHeight > 800 ? windowHeight : undefined }}
            ref={getStartedRef}
        >
            <h1 className={styles.Title} id='get-started'>Get Started</h1>
            <RadioGroup groupName='ColossalAI version:' values={colossalaiVersions} selectedValue={colossalaiVersion} onClick={setColossalaiVersion}></RadioGroup>
            <RadioGroup groupName='PyTorch version:' values={torchVersions} selectedValue={torchVersion} onClick={setTorchVersion}></RadioGroup>
            <RadioGroup groupName='CUDA version:' values={cudaVersions} selectedValue={cudaVersion} onClick={setCudaVersion}></RadioGroup>
            <div>Run this command: <pre>{resolveCommand()}</pre></div>
        </div >
    );
};

export default InstallView;